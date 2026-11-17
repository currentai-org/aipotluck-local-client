"""The CLI home: one copy of this source tree per user, owned by neither installer.

Why it exists: `aipotluck-install` and the desktop GUI each used to keep their own copy of the
tree, so a machine could end up with two CLIs at different versions, and whichever one wrote the
`aipotluck-local-client` shim and the OS service last silently won -- including downgrading to an
older copy. Instead both installers stage what they have into a single neutral folder,
`<layout.install_root>/cli` (the per-user layout only; `--system`/`--install-dir` installs keep
their old behaviour), and the shim and service always point there. Newer wins: a copy is only
replaced when its source commit is *strictly* newer than the installed one, so the order the two
installers happen to run in stops mattering, and uninstalling the GUI cannot take the CLI the
CLI installer put there with it.

Version = the source commit, never a hand-maintained version string. `aipotluck/_build_info.txt`
holds `$Format:%H %ct$` and `.gitattributes` marks it `export-subst`, so anything built with
`git archive` (the GUI's bundled tree, a GitHub tarball) ships the real `<sha> <unix time>`; in a
plain git clone the file stays unsubstituted and `read_build_info` falls back to `git log -1`,
but only when that root really is a checkout (never a repo further up the tree -- a dotfiles repo
in $HOME must never be mistaken for this project's history). Neither source -> sha "unknown",
time 0, which loses to anything already installed.

`current.json` (schema 1) is the manifest: which source directory and interpreter the staged copy
uses, who staged it, and when. This module owns reading and writing both, atomically (write a
temp file, then `os.replace`), so a half-written manifest or build-info file is never observed.

Staging (`stage_source`, `stage_python`) is atomic the same way and one step stronger: the tree is
copied into a hidden `.partial-<pid>` directory and only `os.replace`d into its final
`src-<sha12>` / `python-<name>` name once the copy is complete and carries its `READY_MARKER`. A
copy without the marker is never trusted, so a killed install leaves a partial directory that
`prune` cleans up rather than a half-copied tree that looks installed. `adopt` is the decision
both installers delegate to -- adopt mine, keep the newer one, or do nothing -- and `prune` is the
only thing that deletes staged copies, never the manifest and never a tree holding the running
interpreter.

`register` is the other half of the same idea: the OS service and the `aipotluck-local-client`
shim are pointed at the staged copy's interpreter and source tree, never at whichever checkout or
bundled tree happened to run the install.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from aipotluck.installer import layout
from aipotluck.installer.cli_shim import install_cli_shim
from aipotluck.installer.platform_detect import HostProfile
from aipotluck.installer.service.base import ServiceState, get_service_manager
from aipotluck.proc import NO_WINDOW

log = logging.getLogger("aipotluck.installer.cli_home")

CLI_HOME_DIRNAME = "cli"
MANIFEST_NAME = "current.json"
BUILD_INFO_RELPATH = Path("aipotluck") / "_build_info.txt"
SCHEMA = 1
READY_MARKER = ".complete"
SERVICE_NAME = "aipotluck"

# Never worth copying into the CLI home: history and vendored trees are large, and a staged copy
# with tests/research in it is weight the service and shim never run.
_COPY_IGNORE = shutil.ignore_patterns(
    ".git", "vendor", "tests", "research", "__pycache__", "*.pyc", ".pytest_cache"
)

UNKNOWN_SHA = "unknown"
_GIT_TIMEOUT_S = 10
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class BuildInfo:
    sha: str
    commit_time: int

    def short(self) -> str:
        return self.sha[:12]


@dataclass(frozen=True)
class InstalledCli:
    build: BuildInfo
    src: Path
    python: Path
    installed_by: str
    installed_at: str


def _parse_build_info(text: str) -> BuildInfo | None:
    """`<40-hex sha> <unix seconds>`, or None for anything else -- including the unsubstituted
    `$Format:%H %ct$` a git clone's working tree contains."""
    if text.lstrip().startswith("$Format"):
        return None
    parts = text.split()
    if len(parts) != 2:
        return None
    sha, when = parts
    if not _SHA_RE.match(sha) or not when.isdigit():
        return None
    return BuildInfo(sha, int(when))


def _git_build_info(source_root: Path) -> BuildInfo | None:
    """The commit this checkout is on, or None. Only ever asked about source_root itself: a `.git`
    higher up the tree is a different project (a dotfiles repo in $HOME, say) and must be ignored,
    which is why the caller checks for source_root/.git before this runs."""
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), "log", "-1", "--format=%H %ct"],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_S, **NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return _parse_build_info(result.stdout.strip())


def read_build_info(source_root: Path) -> BuildInfo:
    """Version of the tree at source_root: the substituted `_build_info.txt` if it has one, else
    the checkout's own HEAD, else BuildInfo("unknown", 0). Never raises -- an unknown version is a
    real state (an unpacked copy of a copy), not an error, and it simply never beats an installed
    one."""
    source_root = Path(source_root)
    try:
        text = (source_root / BUILD_INFO_RELPATH).read_text(encoding="utf-8")
        build = _parse_build_info(text)
    except (OSError, UnicodeDecodeError):
        build = None
    if build is not None:
        return build
    if (source_root / ".git").exists():
        fallback = _git_build_info(source_root)
        if fallback is not None:
            return fallback
    return BuildInfo(UNKNOWN_SHA, 0)


def is_newer(candidate: BuildInfo, installed: BuildInfo) -> bool:
    """Strictly newer, by source commit time -- so the same commit installed by the other
    installer never counts as an upgrade, and no install can downgrade."""
    return candidate.commit_time > installed.commit_time


def cli_home(lay: layout.Layout) -> Path:
    return lay.install_root / CLI_HOME_DIRNAME


def read_installed(home: Path) -> InstalledCli | None:
    """What is staged in `home`, or None if nothing usable is: no manifest, unreadable/dead JSON,
    an unknown schema, a missing or mistyped key, or a manifest whose referenced source tree or
    interpreter has since been deleted. Never raises -- None means "nothing to keep", which every
    caller treats as "install fresh"."""
    try:
        data = json.loads((Path(home) / MANIFEST_NAME).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            return None
        if isinstance(data.get("schema"), bool):
            return None
        sha, commit_time = data.get("sha"), data.get("commit_time")
        src, python = data.get("src"), data.get("python")
        installed_by, installed_at = data.get("installed_by"), data.get("installed_at")
        if not isinstance(commit_time, int) or isinstance(commit_time, bool):
            return None
        if not isinstance(sha, str) or not isinstance(src, str) or not isinstance(python, str):
            return None
        if not isinstance(installed_by, str) or not isinstance(installed_at, str):
            return None
        if not (sha and src and python and installed_by and installed_at):
            return None
        src_path, python_path = Path(src), Path(python)
        if not src_path.is_dir() or not python_path.is_file():
            return None
        return InstalledCli(BuildInfo(sha, commit_time), src_path, python_path, installed_by, installed_at)
    except (OSError, ValueError, TypeError, UnicodeDecodeError):
        return None


def write_installed(home: Path, inst: InstalledCli) -> Path:
    """Writes `home/current.json` and returns it. Atomic: a temp file in the same directory then
    `os.replace`, so a crash mid-write leaves the previous manifest intact rather than a truncated
    one -- an interrupted install must not look like a corrupt CLI home."""
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    target = home / MANIFEST_NAME
    tmp = home / f"{MANIFEST_NAME}.tmp-{os.getpid()}"
    payload = {
        "schema": SCHEMA,
        "sha": inst.build.sha,
        "commit_time": inst.build.commit_time,
        "src": str(inst.src),
        "python": str(inst.python),
        "installed_by": inst.installed_by,
        "installed_at": inst.installed_at,
    }
    try:
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, target)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return target


# ------------------------------------------------------------------------------ staging


def _discard(path: Path) -> None:
    """Best-effort removal of a half-built staging directory, for the failure paths."""
    try:
        if path.is_symlink() or not path.is_dir():
            path.unlink()
        else:
            shutil.rmtree(path)
    except OSError:
        pass


def stage_source(source_root: Path, home: Path, build: BuildInfo) -> Path:
    """Copy the tree at `source_root` into `home/src-<sha12>` and return that directory.

    Idempotent: a copy that already carries `READY_MARKER` is reused as is, so reinstalling the
    same commit costs nothing rather than re-copying tens of MB. Otherwise the copy is built under
    a hidden `.src-<sha12>.partial-<pid>` name and only `os.replace`d into place once complete, so
    a killed or failed copy never appears under the name the manifest and the service point at.
    The copied `_build_info.txt` is rewritten with this build's resolved values -- a `git clone`
    copy keeps the unsubstituted `$Format` placeholder, and the staged copy must never have to
    guess its own version from a `.git` it no longer has.
    """
    source_root = Path(source_root)
    home = Path(home)
    dest = home / f"src-{build.short()}"
    if (dest / READY_MARKER).exists():
        return dest
    home.mkdir(parents=True, exist_ok=True)
    partial = home / f".src-{build.short()}.partial-{os.getpid()}"
    try:
        _discard(partial)
        shutil.copytree(source_root, partial, symlinks=True, ignore=_COPY_IGNORE)
        build_info = partial / BUILD_INFO_RELPATH
        build_info.parent.mkdir(parents=True, exist_ok=True)
        build_info.write_text(f"{build.sha} {build.commit_time}\n", encoding="utf-8")
        (partial / READY_MARKER).write_text("", encoding="utf-8")
        _discard(dest)
        os.replace(partial, dest)
    except BaseException:
        _discard(partial)
        raise
    return dest


def stage_python(bundled_dir: Path, home: Path) -> Path:
    """Copy a bundled python-build-standalone directory into `home/python-<name>` and return it.

    Same atomic pattern as `stage_source`, but byte-for-byte: no ignore list, because an
    interpreter's own layout is not ours to second-guess and a missing `lib/` file is a broken
    interpreter. Reused when the marker is already there.
    """
    bundled_dir = Path(bundled_dir)
    home = Path(home)
    dest = home / f"python-{bundled_dir.name}"
    if (dest / READY_MARKER).exists():
        return dest
    home.mkdir(parents=True, exist_ok=True)
    partial = home / f".python-{bundled_dir.name}.partial-{os.getpid()}"
    try:
        _discard(partial)
        shutil.copytree(bundled_dir, partial, symlinks=True)
        (partial / READY_MARKER).write_text("", encoding="utf-8")
        _discard(dest)
        os.replace(partial, dest)
    except BaseException:
        _discard(partial)
        raise
    return dest


def resolve_python(bundled_dir: Path | None, home: Path) -> Path:
    """The interpreter the staged CLI should run under.

    A bundled interpreter is only worth copying when it is the one actually running (the GUI is
    launched by its own bundled python); otherwise the running `sys.executable` is already a
    durable system path and is used as is. `absolute()` rather than `resolve()` on purpose: a
    system `python3` is usually a symlink into a versioned directory, and both the decision and
    the returned path must stay the path the caller invoked, not whatever it points at today.
    """
    executable = Path(sys.executable).absolute()
    if bundled_dir is None:
        return executable
    bundled_path = Path(bundled_dir)
    try:
        rel = executable.relative_to(bundled_path.absolute())
    except ValueError:
        return executable
    return stage_python(bundled_path, home) / rel


# ------------------------------------------------------------------------- the adopt decision


@dataclass(frozen=True)
class AdoptResult:
    """`action` is "adopted" (staged this source), "current" (home already holds this exact
    commit) or "kept-newer" (home holds something newer; nothing was written)."""

    action: str
    installed: InstalledCli


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def adopt(
    source_root: Path,
    lay: layout.Layout,
    *,
    bundled_python_dir: Path | None = None,
    installed_by: str = "cli",
    force: bool = False,
) -> AdoptResult:
    """Stage `source_root` into this user's CLI home unless what is already there is as new.

    The single decision both installers delegate to, which is what makes the order they run in
    stop mattering: the same commit does nothing, an older one never downgrades, and `force` is
    the escape hatch for a deliberate reinstall (or a corrupt home). Nothing is written in the
    "current"/"kept-newer" cases -- not even the manifest.
    """
    home = cli_home(lay)
    candidate = read_build_info(source_root)
    installed = read_installed(home)

    if installed is not None and not force:
        if installed.build.sha == candidate.sha and installed.build.sha != UNKNOWN_SHA:
            log.info(
                "CLI home %s already holds build %s; nothing to stage.", home, candidate.short()
            )
            return AdoptResult("current", installed)
        if not is_newer(candidate, installed.build):
            log.info(
                "CLI home %s holds a newer build (%s at %s) than this one (%s at %s); keeping it.",
                home,
                installed.build.short(),
                installed.build.commit_time,
                candidate.short(),
                candidate.commit_time,
            )
            return AdoptResult("kept-newer", installed)

    src = stage_source(Path(source_root), home, candidate)
    python = resolve_python(bundled_python_dir, home)
    inst = InstalledCli(candidate, src, python, installed_by, _utc_now())
    write_installed(home, inst)
    log.info(
        "Staged CLI build %s (previous: %s) into %s; interpreter %s.",
        candidate.short(),
        installed.build.short() if installed is not None else "none",
        src,
        python,
    )
    return AdoptResult("adopted", inst)


# ------------------------------------------------------- the service and the shim

# Nothing here points at the running checkout: the OS service and the PATH shim are both rewritten
# against the staged copy (InstalledCli.python + InstalledCli.src), which is the whole point of a
# single CLI home -- otherwise a GUI bundled with an older tree, or a checkout that has since been
# deleted, would keep owning the service and the `aipotluck-local-client` command.


def service_args(lay: layout.Layout) -> list[str]:
    """The service's own arguments: where to read runtime.json from and where to log. Identical to
    what install.py has always passed, so a unit registered here is the unit install.py would
    register -- just running the staged copy instead of the checkout."""
    return ["--config-dir", str(lay.config_dir), "--log-dir", str(lay.log_dir)]


def register(
    profile: HostProfile, lay: layout.Layout, inst: InstalledCli, *, register_service: bool, start: bool
) -> tuple[bool, bool, Path]:
    """Points the OS service and the CLI shim at `inst` and returns
    `(service_installed, restarted, shim_path)`.

    `register_service=False` is the case with no service to re-point -- the GUI runs the service
    itself, or nothing was ever installed -- and it must not *create* one: the shim is still
    rewritten either way, because that is what the person's shell actually runs.

    A service that was running is stopped first, restarted afterwards and reported through
    `restarted`; one that was stopped or missing stays that way unless `start` says otherwise, so
    re-pointing a stopped service never silently starts it. A failed registration is an ERROR and a
    failed start a WARNING, not an exception: the staged CLI and the shim are usable regardless,
    and the caller reports the real state rather than an install that "succeeded" into nothing.
    """
    service_installed = False
    restarted = False
    if register_service:
        mgr = get_service_manager(profile.os_name)
        was_running = mgr.status(SERVICE_NAME).state is ServiceState.RUNNING
        if was_running:
            mgr.stop(SERVICE_NAME)
        try:
            mgr.install(
                SERVICE_NAME,
                exec_path=inst.python,
                args=[
                    str(inst.src / "aipotluck" / "service" / "aipotluck_service.py"),
                    *service_args(lay),
                ],
                system_scope=False,
                working_dir=inst.src,
                log_dir=lay.log_dir,
            )
            service_installed = True
        except RuntimeError as exc:
            log.error("%s", exc)
        if service_installed and (start or was_running):
            try:
                mgr.start(SERVICE_NAME, system_scope=False)
                restarted = was_running
            except RuntimeError as exc:
                log.warning("%s", exc)

    shim_path = install_cli_shim(profile, inst.python, False, cli_root=inst.src)
    return service_installed, restarted, shim_path


# ------------------------------------------------------------------------------- pruning


def _is_staged_name(name: str) -> bool:
    """`src-*`/`python-*` copies and their hidden `.partial-<pid>` siblings -- nothing else in the
    CLI home is ours to delete."""
    if name.startswith(("src-", "python-")):
        return True
    return name.startswith((".src-", ".python-")) and ".partial-" in name


def prune(home: Path, keep: Iterable[Path]) -> list[Path]:
    """Delete staged copies in `home` that are neither in `keep` nor holding this process's
    interpreter, and return what was removed.

    Deliberately narrow: only direct children named like staged copies, never the manifest and
    never anything else a user put there. A copy that cannot be removed -- on Windows a running
    interpreter's files are locked -- is a WARNING and the loop continues, because one stuck
    directory must not stop the rest of the cleanup or fail the install that just succeeded.
    """
    home = Path(home)
    keep_paths = {os.path.normcase(os.path.abspath(str(path))) for path in keep}
    executable = Path(sys.executable).absolute()
    removed: list[Path] = []
    try:
        children = sorted(home.iterdir())
    except OSError:
        return removed
    for child in children:
        if not _is_staged_name(child.name):
            continue
        if os.path.normcase(os.path.abspath(str(child))) in keep_paths:
            continue
        try:
            executable.relative_to(child.absolute())
            continue  # the tree holding the running interpreter is never removable
        except ValueError:
            pass
        try:
            if child.is_symlink() or not child.is_dir():
                child.unlink()
            else:
                shutil.rmtree(child)
        except OSError as exc:
            log.warning("Could not remove staged CLI directory %s: %s", child, exc)
            continue
        removed.append(child)
    return removed
