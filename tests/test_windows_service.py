"""aipotluck.installer.service.windows_service's reading of a Scheduled Task's state.

`status()` used to return UNKNOWN for every installed task, so `install` always warned "Service
status is ambiguous" on Windows and nothing could tell a running service from a stopped one. These
read schtasks' real CSV output, captured from a Windows 10 22H2 machine, rather than a mock of it.
"""

from __future__ import annotations

from aipotluck.installer.service.base import ServiceState
from aipotluck.installer.service.windows_service import _task_state


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
