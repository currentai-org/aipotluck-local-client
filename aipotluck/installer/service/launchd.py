"""macOS service backend: launchd.

Default (user/agent scope): ~/Library/LaunchAgents/<label>.plist in the user's GUI domain
(`gui/<uid>`). Starts at login. --system scope: /Library/LaunchDaemons/<label>.plist in the
`system` domain, needs sudo, starts at boot.

The job is KeepAlive, so launchd relaunches the service whenever it exits. That decides how each
operation has to work, and the first version got it wrong (found on a Mac mini, macOS 27):

- **stop** must take the job out of launchd (`bootout`), not ask it to stop. `launchctl stop` on a
  KeepAlive job ends the process and launchd immediately starts a new one, so an uninstall waiting
  for the service to stop waited forever and gave up with "still running".
- **status** must look for a running process (the `PID` in `launchctl list <label>`), not just
  whether launchd knows the job. A loaded job that isn't running was reported as RUNNING.
- **install** must replace a job that's already loaded (bootout, then bootstrap). The legacy
  `launchctl load` leaves the old definition in place, so a reinstall kept running the previous
  interpreter and source paths.
- **install/start** must clear a "disabled" mark first (`launchctl enable`). The legacy
  `unload -w` that the first version uninstalled with writes one permanently, and launchd then
  refuses to bootstrap the label at all ("Bootstrap failed: 5: Input/output error").

`bootstrap`/`bootout`/`kickstart` are launchctl's current interface (macOS 10.10+); `load`/`unload`
are legacy and deprecated. A stopped service stays stopped until `start` or the next login, which
re-reads the plist.
"""

from __future__ import annotations

import logging
import os
import plistlib
import re
import subprocess
import time
from pathlib import Path

from aipotluck.installer.service.base import ServiceManager, ServiceState, ServiceStatus

log = logging.getLogger("aipotluck.installer.service.launchd")

LABEL_PREFIX = "com.aipotluck"

USER_AGENT_DIR = Path.home() / "Library" / "LaunchAgents"
SYSTEM_DAEMON_DIR = Path("/Library/LaunchDaemons")

_PID = re.compile(r'"PID"\s*=\s*(\d+)\s*;')
# How long `stop` waits for launchd to report the job gone after a bootout.
_BOOTOUT_WAIT_SECONDS = 10.0


def _label(name: str) -> str:
    return f"{LABEL_PREFIX}.{name}"


def _plist_path(name: str, system_scope: bool) -> Path:
    plist_dir = SYSTEM_DAEMON_DIR if system_scope else USER_AGENT_DIR
    return plist_dir / f"{_label(name)}.plist"


def _domain(system_scope: bool) -> str:
    return "system" if system_scope else f"gui/{os.getuid()}"


def _target(name: str, system_scope: bool) -> str:
    return f"{_domain(system_scope)}/{_label(name)}"


class LaunchdServiceManager(ServiceManager):
    """See the module docstring for why stop, status and install work the way they do."""

    def install(
        self,
        name: str,
        exec_path: Path,
        args: list[str],
        *,
        system_scope: bool = False,
        working_dir: Path | None = None,
        log_dir: Path | None = None,
    ) -> None:
        plist_path = _plist_path(name, system_scope)
        plist_path.parent.mkdir(parents=True, exist_ok=True)

        program_args = [str(exec_path)] + [str(a) for a in args]
        plist_dict = {
            "Label": _label(name),
            "ProgramArguments": program_args,
            "RunAtLoad": True,
            "KeepAlive": True,
        }
        if working_dir:
            plist_dict["WorkingDirectory"] = str(working_dir)
        if log_dir:
            plist_dict["StandardOutPath"] = str(log_dir / f"{name}.out.log")
            plist_dict["StandardErrorPath"] = str(log_dir / f"{name}.err.log")

        # A job already loaded keeps its old definition until it's booted out; take it out first so
        # the plist written below is the one launchd runs.
        self._bootout(name, system_scope)
        with open(plist_path, "wb") as fh:
            plistlib.dump(plist_dict, fh)
        log.info("Wrote launchd plist: %s", plist_path)
        self._bootstrap(name, system_scope)

    def uninstall(self, name: str, *, system_scope: bool = False) -> None:
        self._bootout(name, system_scope)
        plist_path = _plist_path(name, system_scope)
        if plist_path.exists():
            plist_path.unlink()

    def start(self, name: str, *, system_scope: bool = False) -> None:
        if not self._loaded(name, system_scope):
            # RunAtLoad: bootstrapping starts it.
            self._bootstrap(name, system_scope)
            return
        result = self._launchctl(["kickstart", _target(name, system_scope)])
        if result.returncode != 0:
            raise RuntimeError(f"Couldn't start {_label(name)}: {self._why(result)}")

    def stop(self, name: str, *, system_scope: bool = False) -> None:
        """Takes the job out of launchd so KeepAlive can't relaunch it. The plist stays, so it comes
        back at the next login (or `start`); `uninstall` removes the plist too."""
        if not self._bootout(name, system_scope):
            return
        deadline = time.monotonic() + _BOOTOUT_WAIT_SECONDS
        while self._loaded(name, system_scope) and time.monotonic() < deadline:
            time.sleep(0.2)

    def status(self, name: str, *, system_scope: bool = False) -> ServiceStatus:
        plist_path = _plist_path(name, system_scope)
        result = self._launchctl(["list", _label(name)])
        if result.returncode != 0:
            if not plist_path.exists():
                return ServiceStatus(ServiceState.NOT_INSTALLED, detail=str(plist_path))
            return ServiceStatus(ServiceState.STOPPED, detail="not loaded in launchd")
        pid = _PID.search(result.stdout)
        if pid:
            return ServiceStatus(ServiceState.RUNNING, detail=f"pid {pid.group(1)}")
        return ServiceStatus(ServiceState.STOPPED, detail="loaded, no process running")

    # ------------------------------------------------------------------ launchctl

    def _loaded(self, name: str, system_scope: bool) -> bool:
        return self._launchctl(["list", _label(name)]).returncode == 0

    def _bootstrap(self, name: str, system_scope: bool) -> None:
        plist_path = _plist_path(name, system_scope)
        # A label marked disabled refuses bootstrap ("Bootstrap failed: 5: Input/output error").
        # The legacy `unload -w` sets that mark permanently, so a reinstall on a Mac that
        # ever used it fails. Clear it first.
        self._launchctl(["enable", _target(name, system_scope)])
        result = self._launchctl(["bootstrap", _domain(system_scope), str(plist_path)])
        if result.returncode != 0 and not self._loaded(name, system_scope):
            raise RuntimeError(f"Couldn't load {plist_path} into launchd: {self._why(result)}")

    def _bootout(self, name: str, system_scope: bool) -> bool:
        """True if a loaded job was taken out. A job that wasn't loaded is not an error."""
        if not self._loaded(name, system_scope):
            return False
        result = self._launchctl(["bootout", _target(name, system_scope)])
        if result.returncode != 0 and self._loaded(name, system_scope):
            raise RuntimeError(f"Couldn't stop {_label(name)}: {self._why(result)}")
        return True

    @staticmethod
    def _why(result: subprocess.CompletedProcess) -> str:
        return (result.stderr or result.stdout).strip() or f"launchctl exit {result.returncode}"

    @staticmethod
    def _launchctl(args: list[str]) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(["launchctl"] + args, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("launchctl not found -- not running on macOS?") from exc
