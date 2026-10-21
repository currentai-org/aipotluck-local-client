"""Windows service backend. STUBBED -- not exercised in this environment
(no Windows host to test against).

Default (no admin required): a Scheduled Task via `schtasks`, running at
logon under the current user. This mirrors the no-admin/per-user default
used on Linux (systemd --user) and macOS (LaunchAgents).

--system scope (requires elevation): registers a real Windows Service via
pywin32 (win32serviceutil), started automatically at boot without login --
the true system-service parity with systemd --system / launchd
LaunchDaemons. pywin32 is only required for this path; it's declared as an
optional dependency (see pyproject.toml windows extra) so Linux/macOS
installs never need it.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from aipotluck.installer.service.base import ServiceManager, ServiceState, ServiceStatus

log = logging.getLogger("aipotluck.installer.service.windows_service")

TASK_PREFIX = "AIPotluck"


def _task_name(name: str) -> str:
    return f"{TASK_PREFIX}_{name}"


def _task_state(csv_out: str) -> ServiceState:
    """The task's state from `schtasks /query /fo csv /nh`: one row per trigger, each
    "TaskName","Next Run Time","Status". Running on any row is RUNNING; Ready/Disabled (installed,
    not running right now) is STOPPED. Anything else -- including a localized status word, since
    schtasks prints those in the system's display language -- is UNKNOWN rather than a guess."""
    import csv

    states = set()
    for row in csv.reader(line for line in csv_out.splitlines() if line.strip()):
        if len(row) >= 3:
            states.add(row[2].strip().lower())
    if "running" in states:
        return ServiceState.RUNNING
    if states and states <= {"ready", "disabled", "queued"}:
        return ServiceState.STOPPED
    return ServiceState.UNKNOWN


class WindowsServiceManager(ServiceManager):
    """STUB: no-admin path uses schtasks; --system path uses pywin32.
    Neither has been exercised on real Windows -- implemented per
    documented schtasks/pywin32 conventions only."""

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
        if system_scope:
            self._install_windows_service(name, exec_path, args, working_dir, log_dir)
            return

        task_name = _task_name(name)
        command = " ".join([f'"{exec_path}"'] + [f'"{a}"' for a in args])
        cmd = [
            "schtasks", "/create", "/f",
            "/sc", "onlogon",
            "/tn", task_name,
            "/tr", command,
        ]
        log.info("Registering a logon Scheduled Task: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Surfaced, not swallowed: a task that was never created looks exactly like one that
            # hasn't started yet. "Access is denied" here is usually a non-interactive (SSH,
            # network) logon, whose token Task Scheduler refuses even for a per-user task.
            raise RuntimeError(
                f"Couldn't register the {task_name} Scheduled Task: "
                f"{(result.stderr or result.stdout).strip() or f'schtasks exit {result.returncode}'}"
            )

    def uninstall(self, name: str, *, system_scope: bool = False) -> None:
        if system_scope:
            self._uninstall_windows_service(name)
            return
        subprocess.run(
            ["schtasks", "/delete", "/tn", _task_name(name), "/f"],
            capture_output=True, text=True,
        )

    def start(self, name: str, *, system_scope: bool = False) -> None:
        if system_scope:
            subprocess.run(["sc", "start", name], capture_output=True, text=True)
            return
        subprocess.run(["schtasks", "/run", "/tn", _task_name(name)], capture_output=True, text=True)

    def stop(self, name: str, *, system_scope: bool = False) -> None:
        if system_scope:
            subprocess.run(["sc", "stop", name], capture_output=True, text=True)
            return
        subprocess.run(["schtasks", "/end", "/tn", _task_name(name)], capture_output=True, text=True)

    def status(self, name: str, *, system_scope: bool = False) -> ServiceStatus:
        if system_scope:
            result = subprocess.run(["sc", "query", name], capture_output=True, text=True)
            if result.returncode != 0:
                return ServiceStatus(ServiceState.NOT_INSTALLED, detail=result.stderr.strip())
            if "RUNNING" in result.stdout:
                return ServiceStatus(ServiceState.RUNNING, detail=result.stdout.strip())
            return ServiceStatus(ServiceState.STOPPED, detail=result.stdout.strip())

        result = subprocess.run(
            ["schtasks", "/query", "/tn", _task_name(name), "/fo", "csv", "/nh"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return ServiceStatus(ServiceState.NOT_INSTALLED, detail=result.stderr.strip())
        return ServiceStatus(_task_state(result.stdout), detail=result.stdout.strip())

    @staticmethod
    def _install_windows_service(
        name: str, exec_path: Path, args: list[str],
        working_dir: Path | None, log_dir: Path | None,
    ) -> None:
        """Registers a real Windows Service via pywin32, delegating to
        aipotluck/service/windows_service_host.py (an AipotluckWindowsService /
        win32serviceutil.ServiceFramework subclass). Requires the pywin32
        package (Windows-only optional dependency) and an elevated (admin)
        process. STUB -- implemented per pywin32 docs, not exercised on
        real Windows hardware."""
        try:
            import win32serviceutil  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "pywin32 is required for --system Windows service install. "
                "Install with: pip install aipotluck-local-client[windows]"
            ) from exc

        host_script = Path(__file__).resolve().parent.parent.parent / "service" / "windows_service_host.py"
        python_exe = exec_path  # the resolved python.exe from platform_bootstrap
        cmd = [str(python_exe), str(host_script), "--startup", "auto", "install"]
        log.info("Registering Windows Service (STUB, untested): %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Windows Service registration failed: {result.stderr.strip()}")
        subprocess.run(["sc", "config", name, "start=", "auto"], capture_output=True, text=True)

    @staticmethod
    def _uninstall_windows_service(name: str) -> None:
        subprocess.run(["sc", "delete", name], capture_output=True, text=True)
