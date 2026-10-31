"""Windows service backend.

Default (no admin required): a Scheduled Task that starts the service at THIS user's logon, as
this user. The per-user default, like systemd --user on Linux and a LaunchAgent on macOS. Verified
on Windows 10 22H2, from an interactive desktop session and from an SSH session.

Registered from a Task Scheduler XML definition (`schtasks /create /xml`) rather than
`schtasks /sc onlogon /tr ...`, because:
  - `/sc onlogon` with no user is a logon trigger for ANY user, which needs admin: a standard user
    gets "Access is denied" (seen on a real Windows 10 box). The XML's LogonTrigger names this
    user, which needs no elevation;
  - the user is named by SID, not DOMAIN\name: over SSH %USERDOMAIN% is "WORKGROUP", which
    Task Scheduler can't resolve ("No mapping between account names and security IDs");
  - the XML can turn off Task Scheduler's defaults that would silently stop a long-running service:
    a 72-hour ExecutionTimeLimit, and stop/no-start on battery;
  - it runs `pythonw.exe` (when beside the given python.exe), so no console window opens at logon.

--system scope (requires elevation): registers a real Windows Service via
pywin32 (win32serviceutil), started automatically at boot without login --
the true system-service parity with systemd --system / launchd
LaunchDaemons. pywin32 is only required for this path; it's declared as an
optional dependency (see pyproject.toml windows extra) so Linux/macOS
installs never need it.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import time
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

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


def _current_user_sid() -> str:
    """This process's user SID (S-1-5-21-...), read from its own access token (ctypes: advapi32
    GetTokenInformation + ConvertSidToStringSidW). Not `whoami`: a subprocess can fail for
    environmental reasons that have nothing to do with the user. Raises when it can't be read: a
    task for an unknown principal would be worse than no task."""
    import ctypes
    from ctypes import wintypes

    TOKEN_QUERY = 0x0008
    TokenUser = 1
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    advapi32.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), TOKEN_QUERY, ctypes.byref(token)):
        raise RuntimeError(f"Couldn't open this process's token (error {ctypes.get_last_error()})")
    try:
        needed = wintypes.DWORD()
        advapi32.GetTokenInformation(token, TokenUser, None, 0, ctypes.byref(needed))
        buf = ctypes.create_string_buffer(needed.value)
        if not advapi32.GetTokenInformation(token, TokenUser, buf, needed, ctypes.byref(needed)):
            raise RuntimeError(f"Couldn't read the token's user (error {ctypes.get_last_error()})")
        # TOKEN_USER starts with SID_AND_ATTRIBUTES, whose first field is the PSID.
        psid = ctypes.cast(buf, ctypes.POINTER(ctypes.c_void_p))[0]
        text = wintypes.LPWSTR()
        if not advapi32.ConvertSidToStringSidW(psid, ctypes.byref(text)):
            raise RuntimeError(f"Couldn't format the user's SID (error {ctypes.get_last_error()})")
        try:
            sid = text.value
        finally:
            kernel32.LocalFree(text)
    finally:
        kernel32.CloseHandle(token)
    if not sid or not re.fullmatch(r"S-1-[0-9-]+", sid):
        raise RuntimeError(f"Unexpected SID for the current user: {sid!r}")
    return sid


def _windowless(exec_path: Path) -> Path:
    """pythonw.exe beside python.exe when there is one (python.org and python-build-standalone both
    ship it), so starting the service at logon doesn't open a console window that closes it if the
    person closes the window. Any other interpreter is used as given."""
    if exec_path.name.lower() == "python.exe":
        w = exec_path.with_name("pythonw.exe")
        if w.exists():
            return w
    return exec_path


def task_xml(user_sid: str, exec_path: Path, args: list[str], working_dir: Path | None) -> str:
    """The Task Scheduler definition for the service: start at this user's logon, as this user,
    with no elevation; never time out; keep running on battery; restart if it crashes; one copy."""
    def esc(v: object) -> str:
        return xml_escape(str(v), {'"': "&quot;"})

    arguments = subprocess.list2cmdline([str(a) for a in args])
    workdir = f"\n      <WorkingDirectory>{esc(working_dir)}</WorkingDirectory>" if working_dir else ""
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>aipotluck-local-client: the local inference service (llama-server + tunnel).</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <UserId>{esc(user_sid)}</UserId>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{esc(user_sid)}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>3</Count>
    </RestartOnFailure>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{esc(exec_path)}</Command>
      <Arguments>{esc(arguments)}</Arguments>{workdir}
    </Exec>
  </Actions>
</Task>
"""


class WindowsServiceManager(ServiceManager):
    """No-admin path: a per-user logon Scheduled Task (verified on Windows 10). --system path:
    pywin32, not yet exercised on real Windows."""

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
        xml = task_xml(_current_user_sid(), _windowless(exec_path), args, working_dir)
        xml_path = Path(tempfile.gettempdir()) / f"{task_name}.xml"
        # UTF-16 with a BOM, as schtasks expects for an XML that declares encoding="UTF-16".
        xml_path.write_text(xml, encoding="utf-16")
        cmd = ["schtasks", "/create", "/f", "/tn", task_name, "/xml", str(xml_path)]
        log.info("Registering a per-user logon Scheduled Task: %s", task_name)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        finally:
            xml_path.unlink(missing_ok=True)
        if result.returncode != 0:
            # Surfaced, not swallowed: a task that was never created looks exactly like one that
            # hasn't started yet.
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
        task = _task_name(name)
        result = subprocess.run(["schtasks", "/run", "/tn", task], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Couldn't start the {task} Scheduled Task: {(result.stderr or result.stdout).strip()}")
        # The task runs in the logged-on user's desktop session (an interactive token, so it needs
        # no stored password and no admin). From a session with no desktop -- SSH, a service --
        # schtasks accepts the request and Task Scheduler then quietly doesn't run it ("has not yet
        # run", 0x41303). Say so, rather than letting `login` print "Restarted" over nothing.
        for _ in range(10):
            state = self.status(name).state
            if state is ServiceState.RUNNING:
                return
            time.sleep(0.5)
        raise RuntimeError(
            f"Task Scheduler didn't start {task}. It runs in your desktop session, so it can't be "
            "started from a session without one (such as SSH); it will start at your next sign-in, "
            "or run it from Task Scheduler."
        )

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
        log.info("Registering Windows Service: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Windows Service registration failed: {result.stderr.strip()}")
        subprocess.run(["sc", "config", name, "start=", "auto"], capture_output=True, text=True)

    @staticmethod
    def _uninstall_windows_service(name: str) -> None:
        subprocess.run(["sc", "delete", name], capture_output=True, text=True)
