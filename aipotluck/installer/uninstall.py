"""Uninstall: stop and remove the background service, then delete what the installer put on disk.

    aipotluck-local-client uninstall [--remove-models] [--yes]

What goes, in this order:
1. **The service is stopped** first, so nothing below is in use: llama-server and newt are its
   children and stop with it (on Windows they're in a kill-on-close Job Object, win_job.py).
2. **Models** (only with --remove-models), through llama.cpp's own cache deletion
   (cache_router.delete), never by guessing the HF cache layout. This needs the llama-server binary,
   so it runs before that binary is deleted. A model that can't be deleted stops the uninstall
   there, with everything else left in place, so running it again can finish the job.
3. **The service registration** (systemd unit, launchd plist, or Scheduled Task).
4. **The `aipotluck-local-client` command** (the PATH shim), and on Windows its PATH entry.
5. **This install's directories**: llama.cpp, newt, any source build, config (runtime.json with
   the pairing, presets), logs and the download cache.

What stays: the HF model cache unless asked (it is shared with any other llama.cpp on the machine),
and this checkout of the CLI itself -- whoever installed it removes it (the GUI deletes its own
bundled copy; a CLI install was a `git clone` the user owns).

Deleting the local pairing does not remove the server from the person's aipotluck.org account; that
lives on the server and is removed there.

Every directory deletion is checked first (`_safe_to_delete`): only paths that are inside a
directory named for this app, and never a filesystem root or the home directory, so a bad
--install-dir can't turn this into `rm -rf ~`.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from aipotluck.installer import cache_router, cli_shim, layout as layout_mod
from aipotluck.installer.model_pull import ModelPullError, list_cached_models
from aipotluck.installer.platform_detect import HostProfile
from aipotluck.installer.service.base import ServiceManager, ServiceState

log = logging.getLogger("aipotluck.installer.uninstall")

SERVICE_NAME = "aipotluck"

# Subdirectories of install_root the installer creates. install_root itself goes only if this
# leaves it empty -- with --install-dir it may be a directory the user also keeps other things in.
_INSTALL_SUBDIRS = ("llama.cpp", "llama.cpp-build", "newt", "bin")

_STOP_TIMEOUT_SECONDS = 20.0


class UninstallError(RuntimeError):
    pass


@dataclass
class UninstallReport:
    service_removed: bool = False
    models_removed: list[str] = field(default_factory=list)
    paths_removed: list[str] = field(default_factory=list)
    shim_removed: str | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "service_removed": self.service_removed,
            "models_removed": self.models_removed,
            "paths_removed": self.paths_removed,
            "shim_removed": self.shim_removed,
            "warnings": self.warnings,
        }


def _safe_to_delete(path: Path) -> bool:
    """Only something inside a directory named for this app, and never a root or the home dir.
    The default layouts all put everything under an `aipotluck` directory, and an --install-dir
    override is the one way a caller could point this somewhere else."""
    try:
        resolved = path.resolve()
    except OSError:
        return False
    home = Path.home().resolve()
    if resolved == Path(resolved.anchor) or resolved == home or resolved in home.parents:
        return False
    return layout_mod.APP_NAME in resolved.parts[:-1] or resolved.name == layout_mod.APP_NAME


def _rmtree(path: Path, report: UninstallReport) -> None:
    if not path.exists():
        return
    if not _safe_to_delete(path):
        report.warnings.append(f"left {path} alone: it isn't inside an '{layout_mod.APP_NAME}' directory")
        return

    def make_writable_and_retry(func, p, _exc_info):
        # A read-only file (git's object files on Windows) makes rmtree fail; make it writable and
        # retry once. If that fails too, the error propagates.
        os.chmod(p, 0o700)
        func(p)

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, onerror=make_writable_and_retry)  # onerror: works on 3.9 and later
    else:
        path.unlink()
    report.paths_removed.append(str(path))


def _wait_for_stop(service_mgr: ServiceManager, system_scope: bool) -> bool:
    """True once the OS reports this install's service as not running.

    Asks the service manager, not llama-server's port: another install on the same machine (or the
    desktop user's own, when this runs from a test root) can be serving on that same port, and a
    port can't say whose process it is. Stopping the service stops llama-server and newt with it --
    they're its children (on Windows, in a kill-on-close Job Object)."""
    deadline = time.monotonic() + _STOP_TIMEOUT_SECONDS
    while True:
        try:
            state = service_mgr.status(SERVICE_NAME, system_scope=system_scope).state
        except Exception:  # noqa: BLE001 -- can't tell; don't block the uninstall on it
            return True
        if state is not ServiceState.RUNNING:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.5)


def _remove_windows_path_entry(bin_dir: Path) -> bool:
    """Undo cli_shim's HKCU PATH prepend. Best-effort: False if it couldn't (or wasn't there)."""
    try:
        import winreg
    except ImportError:
        return False
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ | winreg.KEY_WRITE)
    except OSError:
        return False
    try:
        try:
            current, kind = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            return False
        entries = [e for e in current.split(os.pathsep) if e]
        kept = [e for e in entries if os.path.normcase(e.rstrip("\\/")) != os.path.normcase(str(bin_dir))]
        if kept == entries:
            return False
        winreg.SetValueEx(key, "Path", 0, kind, os.pathsep.join(kept))
        return True
    except OSError:
        return False
    finally:
        winreg.CloseKey(key)


def read_llama_cfg(lay: layout_mod.Layout) -> dict:
    try:
        cfg = json.loads((lay.config_dir / "runtime.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    llama = cfg.get("llama_cpp") if isinstance(cfg, dict) else None
    return llama if isinstance(llama, dict) else {}


def uninstall(
    profile: HostProfile,
    lay: layout_mod.Layout,
    service_mgr: ServiceManager,
    *,
    system_scope: bool = False,
    remove_models: bool = False,
    progress=log.info,
) -> UninstallReport:
    report = UninstallReport()
    llama_cfg = read_llama_cfg(lay)

    progress("Stopping the background service")
    try:
        service_mgr.stop(SERVICE_NAME, system_scope=system_scope)
    except Exception as exc:  # noqa: BLE001 -- a service that won't stop is reported, then removed anyway
        report.warnings.append(f"stopping the service: {exc}")
    if not _wait_for_stop(service_mgr, system_scope):
        raise UninstallError(
            "the background service is still running after being asked to stop, so its files are "
            "in use. Nothing was removed; try again in a moment."
        )

    if remove_models:
        server = Path(llama_cfg["server_binary"]) if llama_cfg.get("server_binary") else None
        if server is None or not server.exists():
            raise UninstallError(
                "can't remove downloaded models: llama-server isn't installed, and it's what deletes "
                "them. Nothing was removed."
            )
        try:
            models = list_cached_models(server)
        except ModelPullError as exc:
            raise UninstallError(f"can't list downloaded models ({exc}). Nothing was removed.") from exc
        for model_id in models:
            progress(f"Removing model {model_id}")
            try:
                cache_router.delete(server, model_id)
            except cache_router.CacheRouterError as exc:
                raise UninstallError(
                    f"couldn't remove model {model_id} ({exc}). Removed so far: "
                    f"{', '.join(report.models_removed) or 'nothing'}. Run uninstall again to finish."
                ) from exc
            report.models_removed.append(model_id)

    progress("Removing the background service")
    try:
        service_mgr.uninstall(SERVICE_NAME, system_scope=system_scope)
        report.service_removed = True
    except Exception as exc:  # noqa: BLE001
        report.warnings.append(f"removing the service registration: {exc}")

    bin_dir = cli_shim._bin_dir(profile, system_scope)
    shim = bin_dir / (f"{cli_shim.CLI_SHIM_NAME}.cmd" if profile.os_name == "windows" else cli_shim.CLI_SHIM_NAME)
    if shim.exists():
        shim.unlink()
        report.shim_removed = str(shim)
    if profile.os_name == "windows" and not system_scope:
        _remove_windows_path_entry(bin_dir)
    elif not system_scope:
        for path in cli_shim.remove_from_shell_profiles():
            report.paths_removed.append(f"PATH entry in {path}")

    progress("Deleting downloads, configuration and logs")
    for path in [lay.install_root / sub for sub in _INSTALL_SUBDIRS] + [lay.config_dir, lay.log_dir, lay.state_dir]:
        try:
            _rmtree(path, report)
        except OSError as exc:
            raise UninstallError(
                f"couldn't delete {path} ({exc}). The service is already removed; close whatever "
                "has those files open and run uninstall again to finish."
            ) from exc
    try:
        if lay.install_root.exists() and not any(lay.install_root.iterdir()):
            lay.install_root.rmdir()
            report.paths_removed.append(str(lay.install_root))
    except OSError:
        pass
    return report


def remove_all_models(lay: layout_mod.Layout, progress=log.info) -> list[str]:
    """Delete every downloaded model and leave the install itself alone -- the same path as
    `remove` for each one (model_ops.remove: unload it from the running router, delete it through
    llama.cpp's cache logic, forget its preset and records), so the service keeps running and its
    model list is right afterwards. Stops at the first model that can't be removed."""
    from aipotluck.service import model_ops

    llama_cfg = read_llama_cfg(lay)
    if not llama_cfg.get("server_binary"):
        raise UninstallError("can't remove downloaded models: nothing is installed to remove them with.")
    try:
        models = list_cached_models(Path(llama_cfg["server_binary"]))
    except ModelPullError as exc:
        raise UninstallError(f"can't list downloaded models ({exc})") from exc
    removed: list[str] = []
    for model_id in models:
        progress(f"Removing model {model_id}")
        try:
            model_ops.remove(llama_cfg, lay.config_dir, model_id)
        except model_ops.ModelOpError as exc:
            raise UninstallError(
                f"couldn't remove model {model_id} ({exc}). Removed so far: {', '.join(removed) or 'nothing'}."
            ) from exc
        removed.append(model_id)
    return removed
