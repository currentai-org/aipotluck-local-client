"""aipotluck.installer.model_pull -- `--cache-list` parsing, and the thin wrapper `pull_model`
puts around cache_router's download (whose own real-subprocess tests are in test_cache_router.py).

Deliberately NOT mocked: subprocess spawning. A tiny real stand-in script plays the part of
llama-server instead, so these prove the parsing against a real process's real output.
"""

from __future__ import annotations

import stat
import textwrap
from pathlib import Path

import pytest

from aipotluck.installer import model_pull

FAKE_SERVER_SCRIPT = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import argparse
    import sys
    import time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    parser = argparse.ArgumentParser()
    parser.add_argument("-hf")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--ctx-size")
    parser.add_argument("--gpu-layers")
    parser.add_argument("--fail", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--cache-list", action="store_true")
    parser.add_argument("--cache-list-fail", action="store_true")
    args = parser.parse_args()

    if args.cache_list:
        if args.cache_list_fail:
            print("error: something went wrong reading the cache", file=sys.stderr)
            sys.exit(1)
        print("number of models in cache: 2")
        print("   1. bartowski/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M")
        print("   2. org/other-repo:Q8_0")
        sys.exit(0)

    if args.fail:
        print("error: could not resolve repo/quant", file=sys.stderr)
        sys.exit(1)

    time.sleep(args.delay)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *a):
            pass

    HTTPServer((args.host, args.port), Handler).serve_forever()
    """
)


@pytest.fixture
def fake_server_binary(tmp_path: Path) -> Path:
    script = tmp_path / "fake-llama-server"
    script.write_text(FAKE_SERVER_SCRIPT, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


class TestPullModel:
    def test_a_download_failure_becomes_a_model_pull_error_with_the_reason(self, monkeypatch):
        def fail(binary, target, **kw):
            raise model_pull.cache_router.CacheRouterError("nothing was downloaded for org/nope")

        monkeypatch.setattr(model_pull.cache_router, "download", fail)
        with pytest.raises(model_pull.ModelPullError, match="nothing was downloaded for org/nope"):
            model_pull.pull_model(Path("/fake/llama-server"), "org/nope")

    def test_progress_reaches_the_download(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(model_pull.cache_router, "download", lambda binary, target, **kw: seen.update(kw))
        report = lambda received, total: None  # noqa: E731
        model_pull.pull_model(Path("/fake/llama-server"), "org/repo", timeout=5, on_progress=report)
        assert seen == {"timeout": 5, "on_progress": report}


class TestTerminalProgress:
    class _Stream:
        def __init__(self, tty):
            self.tty, self.written = tty, []

        def isatty(self):
            return self.tty

        def write(self, text):
            self.written.append(text)

        def flush(self):
            pass

    def test_a_tty_gets_one_line_redrawn_in_place(self):
        stream = self._Stream(tty=True)
        report = model_pull.terminal_progress(stream, min_interval=0)
        report(50_000_000, 100_000_000)
        report(100_000_000, 100_000_000)
        assert stream.written[0].startswith("\r") and "50.0%" in stream.written[0]
        assert stream.written[-1].endswith("\n") and "100.0%" in stream.written[-1]

    def test_a_pipe_gets_no_carriage_returns(self, caplog):
        stream = self._Stream(tty=False)
        report = model_pull.terminal_progress(stream)
        with caplog.at_level("INFO", logger="aipotluck.installer.model_pull"):
            for received in range(0, 101, 5):
                report(received * 1_000_000, 100_000_000)
        assert stream.written == []
        assert [r.getMessage().split()[1] for r in caplog.records] == [f"{d}%" for d in range(0, 101, 10)]


class TestListCachedModels:
    def test_parses_real_cache_list_output(self, fake_server_binary):
        models = model_pull.list_cached_models(fake_server_binary)
        assert models == ["bartowski/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M", "org/other-repo:Q8_0"]

    def test_raises_when_binary_missing(self, tmp_path):
        with pytest.raises(model_pull.ModelPullError, match="not found"):
            model_pull.list_cached_models(tmp_path / "does-not-exist")

    def test_raises_when_cache_list_exits_nonzero(self, fake_server_binary, monkeypatch):
        real_run = model_pull.subprocess.run

        def run_with_fail_flag(cmd, *args, **kwargs):
            return real_run(cmd + ["--cache-list-fail"], *args, **kwargs)

        monkeypatch.setattr(model_pull.subprocess, "run", run_with_fail_flag)

        with pytest.raises(model_pull.ModelPullError, match="exited 1"):
            model_pull.list_cached_models(fake_server_binary)

    def test_invokes_exactly_cache_list_no_other_flags(self, fake_server_binary, monkeypatch):
        seen_cmds = []
        real_run = model_pull.subprocess.run

        def recording_run(cmd, *args, **kwargs):
            seen_cmds.append(cmd)
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(model_pull.subprocess, "run", recording_run)

        model_pull.list_cached_models(fake_server_binary)

        assert seen_cmds == [[str(fake_server_binary), "--cache-list"]]
