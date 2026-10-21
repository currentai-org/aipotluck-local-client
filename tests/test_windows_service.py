"""aipotluck.installer.service.windows_service: reading a Scheduled Task's state, and the task
definition the installer registers.

`status()` used to return UNKNOWN for every installed task, so `install` always warned "Service
status is ambiguous" on Windows and nothing could tell a running service from a stopped one. These
read schtasks' real CSV output, captured from a Windows 10 22H2 machine, rather than a mock of it.

The task definition is checked as the XML Task Scheduler will parse (ElementTree), not as text:
the properties that decide whether a standard user can register it, and whether it keeps running.
The real registration is in the Windows-only test at the end.
"""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from aipotluck.installer.service.base import ServiceState
from aipotluck.installer.service.windows_service import WindowsServiceManager, _task_state, task_xml

NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
SID = "S-1-5-21-183394313-3748093834-1322998238-1003"


def test_a_running_task_is_running():
    assert _task_state('"\\AIPotluck_aipotluck","N/A","Running"\n') is ServiceState.RUNNING


def test_an_installed_task_that_is_not_running_is_stopped():
    # Exactly what schtasks /query /fo csv /nh printed for a logon-triggered task on Windows 10.
    assert _task_state('"\\Microsoft\\Windows\\Defrag\\ScheduledDefrag","N/A","Ready"\r\n') is ServiceState.STOPPED
    assert _task_state('"\\AIPotluck_aipotluck","N/A","Disabled"\n') is ServiceState.STOPPED


def test_running_on_any_trigger_row_wins():
    out = '"\\AIPotluck_aipotluck","N/A","Ready"\n"\\AIPotluck_aipotluck","N/A","Running"\n'
    assert _task_state(out) is ServiceState.RUNNING


def test_an_unrecognised_or_localized_status_is_unknown_not_a_guess():
    assert _task_state('"\\AIPotluck_aipotluck","N/A","Bereit"\n') is ServiceState.UNKNOWN
    assert _task_state("") is ServiceState.UNKNOWN
    assert _task_state("garbage") is ServiceState.UNKNOWN


def _parsed(**kw) -> ET.Element:
    args = kw.pop("args", [r"C:\Users\a b\aipotluck\service\aipotluck_service.py", "--config-dir", r"C:\Users\a b\cfg & <x>"])
    return ET.fromstring(task_xml(SID, Path(r"C:\Users\a b\python\pythonw.exe"), args, Path(r"C:\Users\a b\up")))


def test_the_logon_trigger_and_the_principal_are_this_user_so_no_admin_is_needed():
    """`schtasks /sc onlogon` (any user's logon) is refused to a standard user; a trigger scoped to
    this user, run as this user without elevation, is not. Named by SID, which resolves even where
    %USERDOMAIN% is WORKGROUP (an SSH session)."""
    task = _parsed()
    assert task.find("t:Triggers/t:LogonTrigger/t:UserId", NS).text == SID
    principal = task.find("t:Principals/t:Principal", NS)
    assert principal.find("t:UserId", NS).text == SID
    assert principal.find("t:LogonType", NS).text == "InteractiveToken"
    assert principal.find("t:RunLevel", NS).text == "LeastPrivilege"


def test_task_schedulers_defaults_that_would_stop_a_long_running_service_are_off():
    s = _parsed().find("t:Settings", NS)
    assert s.find("t:ExecutionTimeLimit", NS).text == "PT0S"  # default: killed after 72 hours
    assert s.find("t:StopIfGoingOnBatteries", NS).text == "false"  # default: stops when unplugged
    assert s.find("t:DisallowStartIfOnBatteries", NS).text == "false"  # default: won't start unplugged
    assert s.find("t:MultipleInstancesPolicy", NS).text == "IgnoreNew"
    assert s.find("t:RestartOnFailure/t:Count", NS).text == "3"


def test_the_action_runs_the_given_interpreter_with_args_quoted_for_windows():
    """Paths with spaces and XML/shell metacharacters survive both the XML and Windows' own
    command-line parsing (round-tripped through the same rules CreateProcess uses)."""
    exec_ = _parsed().find("t:Actions/t:Exec", NS)
    assert exec_.find("t:Command", NS).text == r"C:\Users\a b\python\pythonw.exe"
    assert exec_.find("t:WorkingDirectory", NS).text == r"C:\Users\a b\up"
    assert exec_.find("t:Arguments", NS).text == (
        r'"C:\Users\a b\aipotluck\service\aipotluck_service.py" --config-dir "C:\Users\a b\cfg & <x>"'
    )


@pytest.mark.skipif(sys.platform != "win32", reason="real Task Scheduler")
def test_a_standard_user_can_register_query_and_remove_the_logon_task(tmp_path, monkeypatch):
    """The real thing, on Windows, as whoever runs the suite (no elevation): register the task the
    installer would, see it Ready, remove it. Under a test name, so a real install is untouched."""
    import aipotluck.installer.service.windows_service as ws

    monkeypatch.setattr(ws, "TASK_PREFIX", "AIPotluckTest")
    mgr = WindowsServiceManager()
    try:
        mgr.install("probe", Path(sys.executable), ["-c", "pass"], working_dir=tmp_path)
        status = mgr.status("probe")
        assert status.state is ServiceState.STOPPED, status
        raw = subprocess.run(["schtasks", "/query", "/tn", "AIPotluckTest_probe", "/xml"],
                             capture_output=True, text=True).stdout
        assert "<LogonTrigger>" in raw and "<ExecutionTimeLimit>PT0S</ExecutionTimeLimit>" in raw
    finally:
        mgr.uninstall("probe")
    assert mgr.status("probe").state is ServiceState.NOT_INSTALLED


def test_a_task_that_task_scheduler_never_runs_is_an_error_not_a_silent_start(monkeypatch):
    """From a session with no desktop (SSH), `schtasks /run` succeeds and the task then never runs
    (Last Result 0x41303, seen on Windows 10). start() must say so, so `login` doesn't report
    'Restarted' over a service that isn't running."""
    import aipotluck.installer.service.windows_service as ws

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        out = '"\\AIPotluck_aipotluck","N/A","Ready"\n' if "/query" in cmd else "SUCCESS"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(ws.subprocess, "run", fake_run)
    monkeypatch.setattr(ws.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="desktop session"):
        WindowsServiceManager().start("aipotluck")
    assert calls[0][:3] == ["schtasks", "/run", "/tn"]


def test_a_task_that_starts_running_is_a_successful_start(monkeypatch):
    import aipotluck.installer.service.windows_service as ws

    states = iter(["Ready", "Running"])

    def fake_run(cmd, **kw):
        out = f'"\\AIPotluck_aipotluck","N/A","{next(states)}"\n' if "/query" in cmd else "SUCCESS"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(ws.subprocess, "run", fake_run)
    monkeypatch.setattr(ws.time, "sleep", lambda s: None)
    WindowsServiceManager().start("aipotluck")  # no exception
