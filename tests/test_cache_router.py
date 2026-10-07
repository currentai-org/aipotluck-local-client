"""cache_router against a real subprocess standing in for llama-server.

The stand-in is a real executable spawned through the real code path, so these prove what a mocked
Popen could not: that the router is actually started, waited on until healthy, sent the request
llama.cpp expects, and torn down afterwards. They do not prove llama.cpp's own behaviour -- that was
verified against the pinned b10989 binary, as recorded in cache_router's docstring.
"""

from __future__ import annotations

import json
import os
import stat
import textwrap
import time
from pathlib import Path

import pytest

from aipotluck.installer import cache_router

FAKE_LLAMA_SERVER = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json, os, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    port = int(sys.argv[sys.argv.index("--port") + 1])
    log_path = os.environ["FAKE_ROUTER_LOG"]

    def record(entry):
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\\n")

    record({"argv": sys.argv[1:], "pid": os.getpid()})

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"status": "ok"})
            self._send(404, {})

        def do_DELETE(self):
            record({"method": "DELETE", "path": self.path})
            message = os.environ.get("FAKE_DELETE_ERROR")
            if message:
                return self._send(500, {"error": {"code": 500, "message": message, "type": "server_error"}})
            self._send(200, {"success": True})

        def log_message(self, *args):
            pass

    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    """
)


@pytest.fixture
def fake_server(tmp_path, monkeypatch):
    binary = tmp_path / "llama-server"
    binary.write_text(FAKE_LLAMA_SERVER, encoding="utf-8")
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    log_path = tmp_path / "router.log"
    monkeypatch.setenv("FAKE_ROUTER_LOG", str(log_path))

    def entries():
        if not log_path.exists():
            return []
        return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

    return binary, entries


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class TestScratchRouter:
    def test_starts_without_a_presets_file(self, fake_server):
        """The whole point: a router with no presets sees every cached model as cache-sourced,
        which is the only kind llama.cpp will delete."""
        binary, entries = fake_server
        with cache_router.scratch_router(binary):
            pass
        argv = entries()[0]["argv"]
        assert "--models-preset" not in argv
        assert argv[argv.index("--host") + 1] == "127.0.0.1"

    def test_the_router_is_stopped_when_the_block_ends(self, fake_server):
        binary, entries = fake_server
        with cache_router.scratch_router(binary):
            pid = entries()[0]["pid"]
            assert _pid_alive(pid)
        deadline = time.monotonic() + 5
        while _pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _pid_alive(pid)

    def test_a_missing_binary_is_an_error_not_a_hang(self, tmp_path):
        with pytest.raises(cache_router.CacheRouterError, match="not found"):
            with cache_router.scratch_router(tmp_path / "nope"):
                pass

    def test_a_router_that_exits_at_startup_is_an_error_not_a_hang(self, tmp_path):
        binary = tmp_path / "llama-server"
        binary.write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")
        binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
        with pytest.raises(cache_router.CacheRouterError, match="code=3"):
            with cache_router.scratch_router(binary, startup_timeout=10):
                pass


class TestDelete:
    def test_sends_a_delete_with_the_model_as_a_url_encoded_query_parameter(self, fake_server):
        """llama.cpp reads the `model` QUERY parameter, not a JSON body; the id contains both "/"
        and ":", so it has to survive encoding intact."""
        binary, entries = fake_server
        cache_router.delete(binary, "org/repo:Q4_K_M")
        requests = [e for e in entries() if e.get("method") == "DELETE"]
        assert requests == [{"method": "DELETE", "path": "/models?model=org%2Frepo%3AQ4_K_M"}]

    def test_a_refusal_raises_with_llama_cpps_own_message(self, fake_server, monkeypatch):
        binary, _entries = fake_server
        monkeypatch.setenv("FAKE_DELETE_ERROR", "model name=org/repo:Q4_K_M is not found")
        with pytest.raises(cache_router.CacheRouterError) as excinfo:
            cache_router.delete(binary, "org/repo:Q4_K_M")
        assert str(excinfo.value) == "model name=org/repo:Q4_K_M is not found"
