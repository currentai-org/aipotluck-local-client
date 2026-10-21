"""Windows: make the service's child processes die with it, however the service ends.

On Linux and macOS the service manager stops the service with SIGTERM, and the service's signal
handler stops llama-server and newt cleanly. Windows has no SIGTERM to deliver to another process:
Task Scheduler's `schtasks /end`, Task Manager's End task, the desktop GUI closing the service it
hosts, and a plain crash all end python.exe with TerminateProcess, which runs no Python code at
all. Every llama-server (the router and each model it loads) and newt it started then keeps running
with no parent -- a tunnel left open, gigabytes of RAM held, and the port taken from the next start.
That was observed on a real Windows 10 machine before this module existed.

The fix is the standard Windows one: the service puts itself in a Job Object configured with
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. Processes it starts afterwards inherit the job. The service's
own process holds the only handle, so when the service ends -- any way at all -- the kernel closes
the handle and kills every process still in the job.

Plain ctypes, so no pywin32. Best-effort: if the job can't be set up (an old Windows without nested
jobs, a parent job that forbids it), it logs a warning and the service runs exactly as before.
"""

from __future__ import annotations

import logging
import sys

log = logging.getLogger("aipotluck.service.win_job")

# winnt.h
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JobObjectExtendedLimitInformation = 9

# Kept for the life of the process: closing it is what kills the job's processes.
_JOB_HANDLE = None


def kill_children_with_this_process() -> bool:
    """Put this process in a kill-on-close job, so everything it starts from now on dies with it.
    True when the job is in place; False (with a warning) when it couldn't be, and on non-Windows."""
    global _JOB_HANDLE
    if sys.platform != "win32":
        return False
    if _JOB_HANDLE is not None:
        return True

    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        log.warning("Couldn't create a Job Object (error %s); child processes may outlive the service.",
                    ctypes.get_last_error())
        return False
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                            ctypes.byref(info), ctypes.sizeof(info)):
        err = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        log.warning("Couldn't configure the Job Object (error %s); child processes may outlive the service.", err)
        return False
    if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
        err = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        log.warning("Couldn't join the Job Object (error %s); child processes may outlive the service.", err)
        return False
    _JOB_HANDLE = job
    log.info("Child processes are tied to this service's lifetime (Windows Job Object).")
    return True
