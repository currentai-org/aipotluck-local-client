"""aipotluck.service.win_job: on Windows, the service's children die with it, however it ends.

Real processes and the real kernel, no mocks: a stand-in "service" joins the job and starts a
long-sleeping child, then is killed the way Windows kills a service (TerminateProcess -- what
schtasks /end, Task Manager and the GUI do, and what runs no Python cleanup). The child must be
gone. Without the job, the same test leaves the child running, which is the orphaned
llama-server/newt this exists to prevent. Windows-only; elsewhere it only checks the no-op.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from aipotluck.service import win_job

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_is_a_no_op_off_windows():
    if sys.platform == "win32":
        pytest.skip("Windows has its own tests below")
    assert win_job.kill_children_with_this_process() is False


def _alive(pid: int) -> bool:
    """Whether `pid` is still running, asked of the kernel directly (tasklist and WMI are refused
    to the non-interactive logons CI and SSH sessions get)."""
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _run_stand_in_service(tmp_path: Path, use_job: bool) -> int:
    """Start a stand-in service that (optionally) joins the job, spawns a child that sleeps for a
    minute, reports the child's pid, then waits. Kill it with TerminateProcess; return the child pid."""
    pid_file = tmp_path / "child.pid"
    script = tmp_path / "service.py"
    script.write_text(textwrap.dedent(f"""
        import subprocess, sys, time
        sys.path.insert(0, {str(REPO_ROOT)!r})
        from aipotluck.service import win_job
        if {use_job!r}:
            assert win_job.kill_children_with_this_process()
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open({str(pid_file)!r}, "w").write(str(child.pid))
        time.sleep(60)
    """))
    service = subprocess.Popen([sys.executable, str(script)])
    deadline = time.time() + 30
    while not pid_file.exists() or not pid_file.read_text():
        assert time.time() < deadline, "stand-in service never started its child"
        time.sleep(0.1)
    child_pid = int(pid_file.read_text())
    assert _alive(child_pid)
    service.kill()  # TerminateProcess on Windows: no signal handler, no atexit, nothing
    service.wait(timeout=10)
    return child_pid


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Objects")
def test_a_killed_service_takes_its_children_with_it(tmp_path):
    child_pid = _run_stand_in_service(tmp_path, use_job=True)
    deadline = time.time() + 10
    while _alive(child_pid) and time.time() < deadline:
        time.sleep(0.2)
    assert not _alive(child_pid), "child outlived the killed service"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Objects")
def test_without_the_job_a_killed_service_orphans_its_children(tmp_path):
    """The failure the job prevents, shown real: so the test above can't pass by accident."""
    child_pid = _run_stand_in_service(tmp_path, use_job=False)
    time.sleep(2)
    try:
        assert _alive(child_pid), "expected the child to be orphaned without the job"
    finally:
        subprocess.run(["taskkill", "/pid", str(child_pid), "/f"], capture_output=True)
