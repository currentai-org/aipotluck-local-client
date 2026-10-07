"""cache_router against a real subprocess standing in for llama-server.

The stand-in is a real executable spawned through the real code path, so these prove what a mocked
Popen could not: that the router is actually started, waited on until healthy, sent the request
llama.cpp expects, and torn down afterwards. They do not prove llama.cpp's own behaviour -- that was
verified against the pinned b10989 binary, as recorded in cache_router's docstring.
"""

from __future__ import annotations

import json
import os
import socket
import stat
import textwrap
import time

import pytest

from aipotluck.installer import cache_router

FAKE_LLAMA_SERVER = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json, os, queue, sys, threading, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    port = int(sys.argv[sys.argv.index("--port") + 1])
    log_path = os.environ["FAKE_ROUTER_LOG"]
    cached = [m for m in os.environ.get("FAKE_CACHED", "").split(",") if m]
    scenario = os.environ.get("FAKE_DOWNLOAD", "ok")
    subscribers = []

    def record(entry):
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\\n")

    def broadcast(event):
        for q in list(subscribers):
            q.put(event)

    def run_download(name):
        if scenario == "hang":
            return
        if scenario == "ok":
            for done in (0, 600, 1200):
                broadcast({"model": "someone/else:Q8_0", "event": "download_progress",
                           "data": {"progress": {"u": {"done": 1, "total": 1}}}})
                broadcast({"model": name, "event": "download_progress", "data": {"progress": {
                    "https://hf/model.gguf": {"done": done, "total": 1200},
                    "https://hf/mmproj.gguf": {"done": done // 4, "total": 300},
                }}})
            cached.append(name)
        if scenario == "phantom":
            # What llama.cpp does for a repo that doesn't exist: logs the error, "finishes" anyway.
            print("0.05.988.397 E get_repo_commit: error: GET failed (401): Invalid username or password.", flush=True)
        broadcast({"model": name, "event": "download_finished"})

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
            if self.path == "/models":
                return self._send(200, {"data": [{"id": m} for m in cached]})
            if self.path == "/models/sse":
                q = queue.Queue()
                subscribers.append(q)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.flush()
                try:
                    while True:
                        self.wfile.write(("data: " + json.dumps(q.get()) + "\\n\\n").encode())
                        self.wfile.flush()
                except OSError:
                    return
            self._send(404, {})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            record({"method": "POST", "path": self.path, "body": body})
            if scenario == "reject":
                return self._send(400, {"error": {"code": 400, "message": "model validation failed, unable to download"}})
            self._send(200, {"success": True})
            threading.Thread(target=lambda: (time.sleep(0.1), run_download(body["model"]))).start()

        def do_DELETE(self):
            record({"method": "DELETE", "path": self.path})
            message = os.environ.get("FAKE_DELETE_ERROR")
            if message:
                return self._send(500, {"error": {"code": 500, "message": message, "type": "server_error"}})
            self._send(200, {"success": True})

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
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


class TestFreePort:
    def test_returns_a_bindable_port(self):
        port = cache_router._free_port("127.0.0.1")
        # If it were still held, a fresh bind on the same port would fail.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))


class TestDownload:
    def test_reports_bytes_across_every_file_and_ignores_other_models(self, fake_server, monkeypatch):
        """A model with a vision projector downloads two files; a progress bar showing only the
        first would hit 100% and then sit there."""
        binary, entries = fake_server
        seen = []
        cache_router.download(binary, "org/repo:Q4_K_M", timeout=15,
                              on_progress=lambda received, total: seen.append((received, total)))
        assert seen == [(0, 1500), (750, 1500), (1500, 1500)]
        posts = [e for e in entries() if e.get("method") == "POST"]
        assert posts == [{"method": "POST", "path": "/models", "body": {"model": "org/repo:Q4_K_M"}}]

    def test_an_already_cached_model_returns_without_downloading(self, fake_server, monkeypatch):
        binary, entries = fake_server
        monkeypatch.setenv("FAKE_CACHED", "org/repo:Q4_K_M")
        cache_router.download(binary, "org/repo:Q4_K_M", timeout=15)
        assert [e for e in entries() if e.get("method") == "POST"] == []

    def test_finished_without_the_model_in_the_cache_is_a_failure_with_llama_cpps_reason(
        self, fake_server, monkeypatch
    ):
        """llama.cpp reports download_finished for a repo that doesn't exist. Taking that at its
        word would report success for a model that isn't there."""
        binary, _entries = fake_server
        monkeypatch.setenv("FAKE_DOWNLOAD", "phantom")
        with pytest.raises(cache_router.CacheRouterError) as excinfo:
            cache_router.download(binary, "nope/nope:Q4_K_M", timeout=15)
        message = str(excinfo.value)
        assert "nothing was downloaded for nope/nope:Q4_K_M" in message
        assert "GET failed (401)" in message

    def test_a_refused_request_raises_with_llama_cpps_own_message(self, fake_server, monkeypatch):
        binary, _entries = fake_server
        monkeypatch.setenv("FAKE_DOWNLOAD", "reject")
        with pytest.raises(cache_router.CacheRouterError, match="model validation failed"):
            cache_router.download(binary, "org/repo:Q4_K_M", timeout=15)

    def test_times_out_and_stops_the_router(self, fake_server, monkeypatch):
        binary, entries = fake_server
        monkeypatch.setenv("FAKE_DOWNLOAD", "hang")
        started = time.monotonic()
        with pytest.raises(cache_router.CacheRouterError, match="timed out"):
            cache_router.download(binary, "org/repo:Q4_K_M", timeout=1)
        assert time.monotonic() - started < 10
        pid = entries()[0]["pid"]
        deadline = time.monotonic() + 5
        while _pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _pid_alive(pid)


class TestTotals:
    def test_total_is_unknown_until_every_file_reports_one(self):
        assert cache_router._totals({"a": {"done": 5, "total": 10}, "b": {"done": 1, "total": 0}}) == (6, None)
