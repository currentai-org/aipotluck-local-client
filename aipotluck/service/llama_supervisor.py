"""Supervises the llama-server child process: start, health-poll, restart
on crash with exponential backoff, and clean shutdown.

This is the layer that satisfies "llama should be running at all times" at
the process level. The OS service manager (systemd/launchd/Windows Service)
supervises *this Python process* the same way; together they give two tiers
of self-healing:

  systemd (Restart=on-failure) -> aipotluck_service.py -> LlamaSupervisor -> llama-server
"""

from __future__ import annotations

import http.client
import json
import logging
import subprocess

from aipotluck.proc import NO_WINDOW
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("aipotluck.service.llama_supervisor")

# Backoff schedule for restart attempts after a crash (seconds).
BACKOFF_SCHEDULE = [1, 2, 5, 10, 20, 30, 60]
# If the process stayed up at least this long, treat the next crash as a
# fresh failure and reset backoff to the start of the schedule.
STABLE_UPTIME_RESET_SECONDS = 120
HEALTH_POLL_INTERVAL_SECONDS = 5
HEALTH_TIMEOUT_SECONDS = 3
STARTUP_HEALTH_TIMEOUT_SECONDS = 60


@dataclass
class LlamaProcessInfo:
    pid: int | None = None
    running: bool = False
    healthy: bool = False
    restart_count: int = 0
    last_exit_code: int | None = None
    last_started_at: float | None = None
    last_error: str = ""


class LlamaSupervisor:
    """Runs llama-server as a monitored child process in a background
    thread. Call start() once; stop() is safe to call multiple times."""

    def __init__(
        self,
        server_binary: Path,
        args: list[str],
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        log_dir: Path | None = None,
        env: dict | None = None,
    ) -> None:
        self.server_binary = server_binary
        self.args = args
        self.host = host
        self.port = port
        self.log_dir = log_dir
        self.env = env

        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._info = LlamaProcessInfo()
        self._log_file = None
        self._terminating = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            log.warning("Supervisor already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="llama-supervisor", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 15.0) -> None:
        self._stop_event.set()
        self._terminate_process(timeout=timeout)
        if self._thread:
            self._thread.join(timeout=timeout + 5)
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def info(self) -> LlamaProcessInfo:
        with self._lock:
            return LlamaProcessInfo(**vars(self._info))

    # -- internals --

    def _run_loop(self) -> None:
        backoff_idx = 0
        while not self._stop_event.is_set():
            started_at = time.monotonic()
            try:
                self._spawn_process()
            except Exception as exc:  # binary missing, permissions, etc.
                log.error("Failed to spawn llama-server: %s", exc)
                with self._lock:
                    self._info.last_error = str(exc)
                if self._stop_event.wait(BACKOFF_SCHEDULE[backoff_idx]):
                    break
                backoff_idx = min(backoff_idx + 1, len(BACKOFF_SCHEDULE) - 1)
                continue

            self._wait_until_healthy_or_stopped()
            exit_code = self._wait_for_exit()
            uptime = time.monotonic() - started_at

            if self._stop_event.is_set():
                break

            with self._lock:
                self._info.running = False
                self._info.healthy = False
                self._info.last_exit_code = exit_code
                self._info.restart_count += 1

            log.warning(
                "llama-server exited (code=%s, uptime=%.1fs); restarting (attempt #%d)",
                exit_code, uptime, self._info.restart_count,
            )

            if uptime >= STABLE_UPTIME_RESET_SECONDS:
                backoff_idx = 0
            delay = BACKOFF_SCHEDULE[backoff_idx]
            log.info("Waiting %ss before restart", delay)
            if self._stop_event.wait(delay):
                break
            backoff_idx = min(backoff_idx + 1, len(BACKOFF_SCHEDULE) - 1)

        log.info("Supervisor loop exiting")

    def _spawn_process(self) -> None:
        cmd = [str(self.server_binary)] + self.args
        log.info("Starting llama-server: %s", " ".join(cmd))

        stdout_target = subprocess.DEVNULL
        stderr_target = subprocess.DEVNULL
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self.log_dir / "llama-server.log", "a", encoding="utf-8")
            stdout_target = self._log_file
            stderr_target = self._log_file

        proc = subprocess.Popen(
            cmd,
            stdout=stdout_target,
            stderr=stderr_target,
            env=self.env,
            **NO_WINDOW,
        )
        with self._lock:
            self._proc = proc
            self._info.pid = proc.pid
            self._info.running = True
            self._info.healthy = False
            self._info.last_started_at = time.time()
            self._info.last_error = ""

    def _wait_until_healthy_or_stopped(self) -> None:
        deadline = time.monotonic() + STARTUP_HEALTH_TIMEOUT_SECONDS
        while time.monotonic() < deadline and not self._stop_event.is_set():
            if self._proc is None or self._proc.poll() is not None:
                return  # process already exited
            if self._check_health():
                with self._lock:
                    self._info.healthy = True
                log.info("llama-server is healthy (pid=%s)", self._info.pid)
                return
            time.sleep(1)
        if not self._stop_event.is_set():
            log.warning("llama-server did not become healthy within %ss", STARTUP_HEALTH_TIMEOUT_SECONDS)

    def _check_health(self) -> bool:
        url = f"http://{self.host}:{self.port}/health"
        try:
            with urllib.request.urlopen(url, timeout=HEALTH_TIMEOUT_SECONDS) as resp:
                return resp.status == 200
        except (urllib.error.URLError, http.client.HTTPException, OSError):
            return False

    def _wait_for_exit(self) -> int | None:
        """Block until the process exits OR stop is requested (in which
        case we terminate it ourselves) OR it becomes unhealthy for too
        long (in which case we restart it)."""
        while True:
            if self._proc is None:
                return None
            try:
                exit_code = self._proc.wait(timeout=HEALTH_POLL_INTERVAL_SECONDS)
                with self._lock:
                    self._proc = None
                return exit_code
            except subprocess.TimeoutExpired:
                pass

            if self._stop_event.is_set():
                self._terminate_process()
                return self._info.last_exit_code

            with self._lock:
                was_healthy = self._info.healthy
            is_healthy = self._check_health()
            with self._lock:
                self._info.healthy = is_healthy
            if was_healthy and not is_healthy:
                log.warning("llama-server health check failed; will keep polling")

    def _terminate_process(self, timeout: float = 15.0) -> None:
        with self._lock:
            proc = self._proc
            already_stopping = self._terminating
            self._terminating = True
        if proc is None or proc.poll() is not None:
            with self._lock:
                self._info.running = False
                self._info.healthy = False
                self._terminating = False
            return
        if already_stopping:
            # Another caller is already terminating this process; just wait.
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
            return
        log.info("Terminating llama-server (pid=%s)", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            log.warning("llama-server did not exit in %ss; killing", timeout)
            proc.kill()
            proc.wait(timeout=5)
        with self._lock:
            self._info.running = False
            self._info.healthy = False
            self._info.last_exit_code = proc.returncode
            self._proc = None
            self._terminating = False
