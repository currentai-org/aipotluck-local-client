"""aipotluck.service.newt_supervisor -- the newt (Pangolin tunnel) process supervisor.

The one thing this file exists to pin: tunnel_connected() correctly distinguishes "the newt
process is alive" from "the tunnel is actually connected" -- process liveness alone can't make
this distinction (a real failure this project hit showed newt "running" for 20+ hours while its
own retry loop never once succeeded), and neither can newt's own --health-file taken alone: it's
a real runtime signal (newt writes "ok" once it has genuinely connected) but confirmed live
against a real Pangolin tunnel that it is NEVER refreshed on later pings and -- the important
part -- is NOT cleared if an already-established connection later drops. So a stale "ok" from an
earlier, now-dead connection would read as healthy forever unless something else catches it; the
log-tail check (kept from this module's first pass) is that catch. Real log/health files are used
throughout (not a mocked reader) -- this is exactly the kind of thing a mock could make look
correct while testing nothing.
"""

from __future__ import annotations

import os
import stat
import textwrap
import time
from pathlib import Path

from aipotluck.service.newt_supervisor import (
    NewtSupervisor,
    _classify_tunnel_state,
    _HEALTH_FILENAME,
    _health_file_says_ok,
    _tail_log_lines,
)


class TestClassifyTunnelState:
    def test_no_lines_is_unknown(self):
        assert _classify_tunnel_state([]) is None

    def test_only_startup_noise_is_unknown(self):
        lines = ["INFO: 2026/09/24 Starting newt version 1.17.0", "INFO: 2026/09/24 Connecting to endpoint..."]
        assert _classify_tunnel_state(lines) is None

    def test_success_line_is_connected(self):
        lines = ["INFO: 2026/09/24 Connecting...", "INFO: 2026/09/24 Tunnel connection to server established successfully!"]
        assert _classify_tunnel_state(lines) is True

    def test_error_line_is_not_connected(self):
        lines = [
            "ERROR: 2026/09/24 14:12:43 Failed to connect: failed to get token: "
            "dial tcp 127.0.0.1:80: connect: connection refused. Retrying in 3s...",
        ]
        assert _classify_tunnel_state(lines) is False

    def test_most_recent_signal_wins_error_after_success(self):
        # A real tunnel that later drops must not keep reporting "connected" forever.
        lines = [
            "INFO: Tunnel connection to server established successfully!",
            "ERROR: 2026/09/24 15:00:00 connection lost, retrying...",
        ]
        assert _classify_tunnel_state(lines) is False

    def test_most_recent_signal_wins_success_after_error(self):
        # The real, common case: newt retries a few times, then succeeds -- must not get stuck
        # reporting the earlier failures forever.
        lines = [
            "ERROR: 2026/09/24 14:12:43 Failed to connect: connection refused. Retrying in 3s...",
            "ERROR: 2026/09/24 14:12:46 Failed to connect: connection refused. Retrying in 3s...",
            "INFO: 2026/09/24 14:12:50 Tunnel connection to server established successfully!",
        ]
        assert _classify_tunnel_state(lines) is True


class TestTailLogLines:
    def test_missing_file_is_empty_not_an_error(self, tmp_path):
        assert _tail_log_lines(tmp_path / "nope.log") == []

    def test_reads_real_file_contents(self, tmp_path):
        path = tmp_path / "newt.log"
        path.write_text("line one\nline two\n", encoding="utf-8")
        assert _tail_log_lines(path) == ["line one", "line two"]

    def test_bounded_by_max_bytes_reads_only_the_tail(self, tmp_path):
        path = tmp_path / "newt.log"
        # Many short lines -- with a small max_bytes, only the last few should come back.
        path.write_text("".join(f"line-{i}\n" for i in range(1000)), encoding="utf-8")
        lines = _tail_log_lines(path, max_bytes=100)
        assert lines[-1] == "line-999"
        assert len(lines) < 20


class TestHealthFileSaysOk:
    def test_none_path_is_false(self):
        assert _health_file_says_ok(None) is False

    def test_missing_file_is_false_not_an_error(self, tmp_path):
        assert _health_file_says_ok(tmp_path / "nope.ok") is False

    def test_matches_real_newt_output(self, tmp_path):
        # Confirmed live: newt's --health-file writes the literal 2-byte content "ok".
        path = tmp_path / "newt-health.ok"
        path.write_text("ok", encoding="utf-8")
        assert _health_file_says_ok(path) is True

    def test_tolerates_trailing_whitespace(self, tmp_path):
        path = tmp_path / "newt-health.ok"
        path.write_text("ok\n", encoding="utf-8")
        assert _health_file_says_ok(path) is True

    def test_wrong_content_is_false(self, tmp_path):
        path = tmp_path / "newt-health.ok"
        path.write_text("not ok", encoding="utf-8")
        assert _health_file_says_ok(path) is False


class TestNewtSupervisorTunnelConnected:
    def test_no_log_dir_is_unknown(self):
        supervisor = NewtSupervisor(
            Path("/bin/true"), tunnel_id="i", tunnel_secret="s", tunnel_endpoint="e", log_dir=None,
        )
        assert supervisor.tunnel_connected() is None

    def test_reads_the_real_configured_log_file(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "newt.log").write_text(
            "ERROR: 2026/09/24 14:12:43 Failed to connect: connection refused. Retrying in 3s...\n",
            encoding="utf-8",
        )
        supervisor = NewtSupervisor(
            Path("/bin/true"), tunnel_id="i", tunnel_secret="s", tunnel_endpoint="e", log_dir=log_dir,
        )
        assert supervisor.tunnel_connected() is False

    def test_info_includes_tunnel_connected_without_affecting_running(self, tmp_path):
        # The exact discrepancy this feature exists to surface: running=True (set directly on
        # the internal info, as _spawn_process would after a real start) alongside a log that
        # shows the tunnel was never actually established.
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "newt.log").write_text(
            "ERROR: 2026/09/24 14:12:43 Failed to connect: connection refused. Retrying in 3s...\n",
            encoding="utf-8",
        )
        supervisor = NewtSupervisor(
            Path("/bin/true"), tunnel_id="i", tunnel_secret="s", tunnel_endpoint="e", log_dir=log_dir,
        )
        supervisor._info.running = True
        supervisor._info.pid = 12345

        info = supervisor.info()

        assert info.running is True
        assert info.pid == 12345
        assert info.tunnel_connected is False

    def test_info_reflects_a_real_successful_connection(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "newt.log").write_text(
            "INFO: 2026/09/24 14:12:50 Tunnel connection to server established successfully!\n",
            encoding="utf-8",
        )
        supervisor = NewtSupervisor(
            Path("/bin/true"), tunnel_id="i", tunnel_secret="s", tunnel_endpoint="e", log_dir=log_dir,
        )
        supervisor._info.running = True

        assert supervisor.info().tunnel_connected is True

    def _supervisor(self, tmp_path) -> tuple[NewtSupervisor, Path, Path]:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        supervisor = NewtSupervisor(
            Path("/bin/true"), tunnel_id="i", tunnel_secret="s", tunnel_endpoint="e", log_dir=log_dir,
        )
        return supervisor, log_dir / "newt.log", log_dir / _HEALTH_FILENAME

    def test_health_file_ok_with_no_log_evidence_is_connected(self, tmp_path):
        # The real, common case right after a fresh connect: the health file is the only signal.
        supervisor, _newt_log, health_file = self._supervisor(tmp_path)
        health_file.write_text("ok", encoding="utf-8")
        assert supervisor.tunnel_connected() is True

    def test_stale_ok_health_file_is_overridden_by_a_fresh_error(self, tmp_path):
        # The exact bug this feature exists to fix, reproduced directly: confirmed live that newt
        # does NOT clear --health-file when an already-established connection later drops, so a
        # stale "ok" alone would read as healthy forever. The log's fresh ERROR must win.
        supervisor, newt_log, health_file = self._supervisor(tmp_path)
        health_file.write_text("ok", encoding="utf-8")
        newt_log.write_text(
            "ERROR: 2026/09/24 15:00:00 Failed to connect: connection refused. Retrying in 3s...\n",
            encoding="utf-8",
        )
        assert supervisor.tunnel_connected() is False

    def test_malformed_health_file_falls_through_to_log_signal(self, tmp_path):
        supervisor, newt_log, health_file = self._supervisor(tmp_path)
        health_file.write_text("not ok", encoding="utf-8")
        newt_log.write_text(
            "INFO: Tunnel connection to server established successfully!\n", encoding="utf-8",
        )
        assert supervisor.tunnel_connected() is True

    def test_no_health_file_and_no_log_evidence_is_unknown(self, tmp_path):
        supervisor, _newt_log, _health_file = self._supervisor(tmp_path)
        assert supervisor.tunnel_connected() is None


FAKE_NEWT_SCRIPT = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import os
    import sys
    import time

    argv_log = os.environ.get("FAKE_NEWT_ARGV_LOG")
    if argv_log:
        with open(argv_log, "w", encoding="utf-8") as fh:
            fh.write(" ".join(sys.argv[1:]))
    env_log = os.environ.get("FAKE_NEWT_ENV_LOG")
    if env_log:
        with open(env_log, "w", encoding="utf-8") as fh:
            fh.write(os.environ.get("NEWT_SECRET", ""))
    time.sleep(30)
    """
)


class TestSpawnProcessHealthFileWiring:
    """Real subprocess, not a mocked one -- proves _spawn_process actually passes --health-file
    on the real command line and actually clears stale state on disk, not just that the code
    reads that way."""

    def _make_fake_newt(self, tmp_path: Path) -> Path:
        script = tmp_path / "fake_newt.py"
        script.write_text(FAKE_NEWT_SCRIPT, encoding="utf-8")
        script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return script

    def test_health_file_flag_is_passed_on_the_real_command_line(self, tmp_path):
        fake_newt = self._make_fake_newt(tmp_path)
        log_dir = tmp_path / "logs"
        argv_log = tmp_path / "argv.txt"
        supervisor = NewtSupervisor(
            fake_newt, tunnel_id="i", tunnel_secret="s", tunnel_endpoint="e", log_dir=log_dir,
            env={**os.environ, "FAKE_NEWT_ARGV_LOG": str(argv_log)},
        )
        try:
            supervisor._spawn_process()
            deadline = time.monotonic() + 5
            while not argv_log.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            content = argv_log.read_text(encoding="utf-8")
        finally:
            supervisor._terminate_process(timeout=5)

        assert "--health-file" in content
        assert str(log_dir / _HEALTH_FILENAME) in content

    def test_secret_is_passed_in_the_environment_and_never_on_the_command_line(self, tmp_path):
        fake_newt = self._make_fake_newt(tmp_path)
        argv_log = tmp_path / "argv.txt"
        env_log = tmp_path / "env.txt"
        supervisor = NewtSupervisor(
            fake_newt, tunnel_id="the-id", tunnel_secret="s3cr3t-value", tunnel_endpoint="e",
            log_dir=tmp_path / "logs",
            env={**os.environ, "FAKE_NEWT_ARGV_LOG": str(argv_log), "FAKE_NEWT_ENV_LOG": str(env_log)},
        )
        try:
            supervisor._spawn_process()
            deadline = time.monotonic() + 5
            while not (argv_log.exists() and env_log.exists()) and time.monotonic() < deadline:
                time.sleep(0.05)
            argv = argv_log.read_text(encoding="utf-8")
            secret_from_env = env_log.read_text(encoding="utf-8")
        finally:
            supervisor._terminate_process(timeout=5)

        assert "s3cr3t-value" not in argv
        assert "--secret" not in argv
        assert "--id the-id" in argv
        assert secret_from_env == "s3cr3t-value"

    def test_stale_health_file_is_removed_before_a_fresh_spawn(self, tmp_path):
        fake_newt = self._make_fake_newt(tmp_path)
        log_dir = tmp_path / "logs"
        log_dir.mkdir(parents=True)
        stale = log_dir / _HEALTH_FILENAME
        stale.write_text("ok", encoding="utf-8")
        supervisor = NewtSupervisor(
            fake_newt, tunnel_id="i", tunnel_secret="s", tunnel_endpoint="e", log_dir=log_dir,
        )
        try:
            supervisor._spawn_process()
            assert not stale.exists()
        finally:
            supervisor._terminate_process(timeout=5)
