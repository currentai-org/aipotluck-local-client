"""Real Windows Service host for aipotluck (--system, admin-only path).

Wraps the same AipotluckServiceRunner used everywhere else (Linux systemd,
macOS launchd, Windows Scheduled Task) inside a pywin32 ServiceFramework,
so llama-server supervision can run as a proper Windows Service -- started
at boot, no login required, restarted by the SCM on crash.

STUB STATUS: written against documented pywin32 APIs
(https://github.com/mhammond/pywin32) and **not yet run on real Windows
hardware** -- unlike the default per-user path, a Scheduled Task, which
has been. Requires:
    pip install aipotluck-local-client[windows]

CLI, driven by aipotluck/installer/service/windows_service.py:
    python windows_service_host.py install
    python windows_service_host.py start
    python windows_service_host.py stop
    python windows_service_host.py remove
Or run with no args when launched by the Windows SCM itself.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aipotluck.service.runner import DEFAULT_HOST, DEFAULT_PORT, SERVICE_NAME, AipotluckServiceRunner  # noqa: E402

try:
    import servicemanager  # type: ignore
    import win32event  # type: ignore
    import win32service  # type: ignore
    import win32serviceutil  # type: ignore
except ImportError as exc:  # only importable on Windows with pywin32 installed
    raise ImportError(
        "pywin32 is required to run aipotluck as a Windows Service. "
        "Install with: pip install aipotluck-local-client[windows]"
    ) from exc

log = logging.getLogger("aipotluck.service.windows_host")

DEFAULT_SYSTEM_CONFIG_DIR = Path(r"C:\ProgramData\aipotluck\config")
DEFAULT_SYSTEM_LOG_DIR = Path(r"C:\ProgramData\aipotluck\logs")


def _setup_logging(log_dir: Path | None) -> None:
    handlers: list[logging.Handler] = []
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                log_dir / f"{SERVICE_NAME}.log", maxBytes=5 * 1024 * 1024, backupCount=3
            )
        )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers or None,
    )


class AipotluckWindowsService(win32serviceutil.ServiceFramework):
    """Implemented per pywin32's documented ServiceFramework contract, not
    yet run on real Windows. SvcDoRun/SvcStop are the two SCM callbacks;
    everything else is delegated to the same AipotluckServiceRunner used by
    every other OS."""

    _svc_name_ = SERVICE_NAME
    _svc_display_name_ = "aipotluck local inference client"
    _svc_description_ = "Runs and supervises the local llama.cpp inference server for aipotluck."

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)

        config_dir = Path(os.environ.get("AIPOTLUCK_CONFIG_DIR", str(DEFAULT_SYSTEM_CONFIG_DIR)))
        log_dir = Path(os.environ.get("AIPOTLUCK_LOG_DIR", str(DEFAULT_SYSTEM_LOG_DIR)))
        _setup_logging(log_dir)

        self.runner = AipotluckServiceRunner(DEFAULT_HOST, DEFAULT_PORT, config_dir, log_dir)

    def SvcStop(self) -> None:
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        log.info("SvcStop requested")
        self.runner.stop()
        win32event.SetEvent(self.stop_event)

    def SvcDoRun(self) -> None:
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, ""),
        )
        log.info("SvcDoRun starting")
        self.runner.start()
        win32event.WaitForSingleObject(self.stop_event, win32event.INFINITE)
        log.info("SvcDoRun exiting")


def main() -> None:
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(AipotluckWindowsService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(AipotluckWindowsService)


if __name__ == "__main__":
    main()
