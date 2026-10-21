#!/usr/bin/env python3
"""aipotluck-local-client service entry point (CLI/foreground form).

This is the exact command invoked by:
  - systemd (Linux, --user or --system unit)
  - launchd (macOS, LaunchAgent or LaunchDaemon)
  - Scheduled Task (Windows, no-admin default: schtasks /sc onlogon)

All OS-specific process supervision (auto-restart, run-at-login/boot) is
handled by those service managers wrapping this single, portable script --
this file itself contains no OS-specific code. The heavy lifting lives in
aipotluck/service/runner.py (AipotluckServiceRunner) and
aipotluck/service/llama_supervisor.py.

For a real Windows Service (--system, requires elevation), see
aipotluck/service/windows_service_host.py, which wraps AipotluckServiceRunner
in a pywin32 ServiceFramework instead of this signal-driven CLI form.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import signal
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aipotluck.service.runner import DEFAULT_HOST, DEFAULT_PORT, SERVICE_NAME, AipotluckServiceRunner  # noqa: E402
from aipotluck.service.win_job import kill_children_with_this_process  # noqa: E402

log = logging.getLogger("aipotluck.service")


def default_config_dir() -> Path:
    """Best-effort default when --config-dir isn't passed. The installer
    always passes --config-dir explicitly, so this only matters for a
    manual/ad-hoc launch."""
    if env_dir := os.environ.get("AIPOTLUCK_CONFIG_DIR"):
        return Path(env_dir)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "aipotluck" / "config"
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(local_appdata) / "aipotluck" / "config"
    return Path.home() / ".config" / "aipotluck"


def setup_logging(log_dir: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{SERVICE_NAME}.log", maxBytes=5 * 1024 * 1024, backupCount=3
        )
        handlers.append(file_handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


def run_foreground(host: str, port: int, config_dir: Path, log_dir: Path | None) -> None:
    # Before anything is spawned, so llama-server and newt inherit it (Windows only; a no-op
    # elsewhere). Without it, any stop on Windows -- schtasks /end, a crash, the GUI closing --
    # orphans them, because none of those runs the signal handler below.
    kill_children_with_this_process()
    runner = AipotluckServiceRunner(host, port, config_dir, log_dir)
    runner.start()

    def handle_signal(signum, _frame):
        log.info("Received signal %s, shutting down", signum)
        # runner.stop() joins the HTTP server thread and waits (with a
        # timeout) for llama-server to terminate -- both bounded operations,
        # safe to run directly on a Python signal handler since neither
        # blocks on this same thread's own event loop (unlike calling
        # http.server's shutdown() from inside serve_forever() itself,
        # which is why the server now runs on its own thread in
        # AipotluckServiceRunner rather than on the main thread).
        runner.stop()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, "SIGBREAK"):  # Windows Ctrl+Break, when run interactively
        signal.signal(signal.SIGBREAK, handle_signal)  # type: ignore[attr-defined]

    runner.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="aipotluck local service")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--config-dir", type=Path, default=default_config_dir())
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    setup_logging(args.log_dir)
    run_foreground(args.host, args.port, args.config_dir, args.log_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
