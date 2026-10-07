"""Supervises the `newt` (Pangolin tunnel client) child process: start,
restart on crash with exponential backoff, and clean shutdown.

Cloned from llama_supervisor.py's shape (same rationale: the OS service
manager supervises this Python process, this class supervises the child
process it owns -- two tiers of self-healing).

  systemd (Restart=on-failure) -> aipotluck_service.py -> NewtSupervisor -> newt

Unlike llama-server, newt has no local HTTP `/health` to poll -- but as of newt 1.17.0 it does
have a real runtime health signal: `--health-file <path>`, which newt writes itself once it has
genuinely established the tunnel. Confirmed live, against a real local Pangolin stack, exactly how
this behaves (not assumed from the --help text):
  - newt actively removes any stale health file at its OWN process startup, before it has
    reconnected -- so a freshly (re)started process never inherits a leftover "ok" from before.
  - Once connected, it writes the literal 2-byte content "ok".
  - It is written ONCE on connect and is NEVER refreshed on later pings, and -- this is the part
    worth stating plainly, because it's the opposite of what the flag name suggests -- newt does
    NOT remove or update it again if an already-established connection later drops. Verified by
    killing the real Pangolin server under a connected newt process: it went straight into its
    own `ERROR:`-logged retry loop while the health file kept saying "ok" from the earlier,
    now-stale connection.

So the health file alone answers "did this process connect successfully at least once since it
started" -- a real, positive, runtime-computed signal, but not by itself "is it connected RIGHT
NOW". That's why tunnel_connected() below combines it with the log-tail check this module already
had: newt's own log keeps emitting a fresh ERROR line for as long as it's genuinely still failing
to (re)connect, so a recent ERROR line is treated as authoritative and overrides a stale "ok" --
the log-tail's role is now specifically that override, not an independent positive claim.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("aipotluck.service.newt_supervisor")

BACKOFF_SCHEDULE = [1, 2, 5, 10, 20, 30, 60]
STABLE_UPTIME_RESET_SECONDS = 120
POLL_INTERVAL_SECONDS = 5

# newt's own log line for a real, successful control-channel connection -- confirmed against a
# real Pangolin tunnel during this project's original CUR-1266 validation (see
# aipotluck.org/pangolin/FAQ.md's e2e-test section: "look for 'Tunnel connection to server
# established successfully!'"). newt logs a fresh "ERROR:"-prefixed line every retry for as long
# as it's genuinely failing to connect (its own internal retry loop, independent of whether the
# OS process itself is alive) -- see this module's docstring for why this is now used only as an
# override signal for a stale health file, not an independent positive claim.
_TUNNEL_CONNECTED_SIGNAL = "established successfully"
_TUNNEL_ERROR_PREFIX = "ERROR:"
_LOG_TAIL_BYTES = 16 * 1024

# newt's own --health-file content when genuinely connected -- confirmed live, not assumed from
# docs (see module docstring). Compared after stripping whitespace/newlines defensively; the
# exact byte content isn't documented as a stability guarantee.
_HEALTH_FILE_OK_CONTENT = "ok"
_HEALTH_FILENAME = "newt-health.ok"

# newt reads its secret from this environment variable as well as from `--secret`, and the
# environment wins over its own config file (both confirmed against newt 1.18.1's -show-config).
# The difference matters: any local user can read another process's argv with `ps`, but only the
# same user or root can read its environment.
_SECRET_ENV_VAR = "NEWT_SECRET"


@dataclass
class NewtProcessInfo:
    pid: int | None = None
    running: bool = False
    restart_count: int = 0
    last_exit_code: int | None = None
    last_started_at: float | None = None
    last_error: str = ""
    # None: no signal yet from either check (fresh start, nothing logged, no health file written).
    # True/False: see NewtSupervisor.tunnel_connected()'s docstring for how the real runtime
    # --health-file check and the log-tail override combine to produce this.
    # Deliberately NOT folded into `running` -- a running-but-never-connected newt process is
    # exactly the failure mode this field exists to make visible, not hide behind "running: true".
    tunnel_connected: bool | None = None


class NewtSupervisor:
    """Runs `newt` as a monitored child process in a background thread.
    Call start() once; stop() is safe to call multiple times."""

    def __init__(
        self,
        newt_binary: Path,
        *,
        tunnel_id: str,
        tunnel_secret: str,
        tunnel_endpoint: str,
        log_dir: Path | None = None,
        env: dict | None = None,
    ) -> None:
        self.newt_binary = newt_binary
        self.tunnel_id = tunnel_id
        self.tunnel_secret = tunnel_secret
        self.tunnel_endpoint = tunnel_endpoint
        self.log_dir = log_dir
        self.env = env

        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._info = NewtProcessInfo()
        self._log_file = None
        self._terminating = False

    @property
    def health_file_path(self) -> Path | None:
        """None when log_dir wasn't given -- mirrors newt.log's own gating (see
        _spawn_process): there's nowhere sensible to put it without one."""
        return self.log_dir / _HEALTH_FILENAME if self.log_dir else None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            log.warning("Newt supervisor already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="newt-supervisor", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 15.0) -> None:
        self._stop_event.set()
        self._terminate_process(timeout=timeout)
        if self._thread:
            self._thread.join(timeout=timeout + 5)
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def info(self) -> NewtProcessInfo:
        with self._lock:
            info = NewtProcessInfo(**vars(self._info))
        info.tunnel_connected = self.tunnel_connected()
        return info

    def tunnel_connected(self) -> bool | None:
        """Combines newt's own --health-file (a real runtime signal newt computes from its actual
        connection state, not text-pattern matching) with a log-tail check, in that priority
        order. Computed fresh every call (never cached in self._info) -- it has to reflect newt's
        CURRENT state, not a snapshot from whenever it last transitioned.

        Why both are needed, confirmed live against a real Pangolin tunnel (see module
        docstring): the health file says "ok" once connected but is NEVER updated again on later
        pings, and -- critically -- is NOT cleared if an already-established connection later
        drops. So a stale "ok" from an earlier, now-dead connection would read as healthy forever
        if trusted alone. The log-tail check closes that gap: newt keeps emitting a fresh ERROR
        line for as long as it's genuinely still failing to (re)connect, so a recent ERROR line
        overrides a stale "ok". A log line that itself claims success is kept as a fallback
        positive signal (in case the health file write is momentarily behind, or --health-file
        was ever unavailable for some reason) rather than an independent source of truth.
        """
        if not self.log_dir:
            return None

        log_state = _classify_tunnel_state(_tail_log_lines(self.log_dir / "newt.log"))
        if log_state is False:
            return False  # a fresh ERROR line always wins -- overrides a stale "ok" health file

        if _health_file_says_ok(self.health_file_path):
            return True

        return log_state  # True (log itself claims success) or None (no evidence yet)

    # -- internals --

    def _run_loop(self) -> None:
        backoff_idx = 0
        while not self._stop_event.is_set():
            started_at = time.monotonic()
            try:
                self._spawn_process()
            except Exception as exc:  # binary missing, permissions, etc.
                log.error("Failed to spawn newt: %s", exc)
                with self._lock:
                    self._info.last_error = str(exc)
                if self._stop_event.wait(BACKOFF_SCHEDULE[backoff_idx]):
                    break
                backoff_idx = min(backoff_idx + 1, len(BACKOFF_SCHEDULE) - 1)
                continue

            exit_code = self._wait_for_exit()
            uptime = time.monotonic() - started_at

            if self._stop_event.is_set():
                break

            with self._lock:
                self._info.running = False
                self._info.last_exit_code = exit_code
                self._info.restart_count += 1

            log.warning(
                "newt exited (code=%s, uptime=%.1fs); restarting (attempt #%d)",
                exit_code, uptime, self._info.restart_count,
            )

            if uptime >= STABLE_UPTIME_RESET_SECONDS:
                backoff_idx = 0
            delay = BACKOFF_SCHEDULE[backoff_idx]
            log.info("Waiting %ss before restart", delay)
            if self._stop_event.wait(delay):
                break
            backoff_idx = min(backoff_idx + 1, len(BACKOFF_SCHEDULE) - 1)

        log.info("Newt supervisor loop exiting")

    def _spawn_process(self) -> None:
        # The secret goes in the environment, never on the command line -- see _SECRET_ENV_VAR.
        cmd = [
            str(self.newt_binary),
            "--id", self.tunnel_id,
            "--endpoint", self.tunnel_endpoint,
        ]
        env = {**(self.env if self.env is not None else os.environ), _SECRET_ENV_VAR: self.tunnel_secret}
        log.info(
            "Starting newt: %s --id %s --endpoint %s (secret passed via %s)",
            self.newt_binary, self.tunnel_id, self.tunnel_endpoint, _SECRET_ENV_VAR,
        )

        stdout_target = subprocess.DEVNULL
        stderr_target = subprocess.DEVNULL
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self.log_dir / "newt.log", "a", encoding="utf-8")
            stdout_target = self._log_file
            stderr_target = self._log_file

            health_file = self.log_dir / _HEALTH_FILENAME
            # Belt and suspenders: newt tries to clear its own stale health file at startup
            # (confirmed live), but don't rely on that being every version's behavior -- a leftover
            # "ok" from a previous process that this fresh spawn hasn't earned yet would be read as
            # healthy immediately, before newt has reconnected to anything.
            health_file.unlink(missing_ok=True)
            cmd += ["--health-file", str(health_file)]

        proc = subprocess.Popen(
            cmd,
            stdout=stdout_target,
            stderr=stderr_target,
            env=env,
        )
        with self._lock:
            self._proc = proc
            self._info.pid = proc.pid
            self._info.running = True
            self._info.last_started_at = time.time()
            self._info.last_error = ""

    def _wait_for_exit(self) -> int | None:
        """Block until the process exits OR stop is requested (in which case we terminate it
        ourselves). No health check to poll -- "running" IS "healthy" here, unlike
        LlamaSupervisor, since newt exposes no local HTTP endpoint of its own."""
        while True:
            if self._proc is None:
                return None
            try:
                exit_code = self._proc.wait(timeout=POLL_INTERVAL_SECONDS)
                with self._lock:
                    self._proc = None
                return exit_code
            except subprocess.TimeoutExpired:
                pass

            if self._stop_event.is_set():
                self._terminate_process()
                return self._info.last_exit_code

    def _terminate_process(self, timeout: float = 15.0) -> None:
        with self._lock:
            proc = self._proc
            already_stopping = self._terminating
            self._terminating = True
        if proc is None or proc.poll() is not None:
            with self._lock:
                self._info.running = False
                self._terminating = False
            return
        if already_stopping:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
            return
        log.info("Terminating newt (pid=%s)", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            log.warning("newt did not exit in %ss; killing", timeout)
            proc.kill()
            proc.wait(timeout=5)
        with self._lock:
            self._info.running = False
            self._info.last_exit_code = proc.returncode
            self._proc = None
            self._terminating = False


def _tail_log_lines(path: Path, max_bytes: int = _LOG_TAIL_BYTES) -> list[str]:
    """The last chunk of a log file as lines -- tolerant of it not existing yet (fresh start,
    nothing logged) or being briefly unreadable, and deliberately bounded so this stays cheap to
    call on every /status request even once the file has been growing for days."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            data = fh.read()
    except OSError:
        return []
    return data.decode("utf-8", errors="replace").splitlines()


def _classify_tunnel_state(lines: list[str]) -> bool | None:
    """Scan from the end for the first line that says anything -- the most recent signal wins,
    since newt keeps re-logging a fresh ERROR line for as long as it's genuinely still failing to
    connect, and only logs the success line once it actually has."""
    for line in reversed(lines):
        if _TUNNEL_CONNECTED_SIGNAL in line:
            return True
        if _TUNNEL_ERROR_PREFIX in line:
            return False
    return None


def _health_file_says_ok(path: Path | None) -> bool:
    """True only if the file exists and its content matches newt's real --health-file output.
    Missing (never written, or removed at newt's own startup before reconnecting) is False, not
    an error -- that's the expected state for "not connected yet"."""
    if path is None:
        return False
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return content == _HEALTH_FILE_OK_CONTENT
