"""Shared fixtures for the aipotluck-local-client test suite.

Nothing here touches the real host: no real HOME, no real systemd/launchd, no real network. Tests
that need a "Layout" get one rooted under `tmp_path`; tests that need a service manager or fetch
call get a fake/mock instead of the real thing.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# scripts/ holds standalone tools (package_custom_build.py and, per its own docstring, more to
# come) that aren't part of the aipotluck package -- put the directory itself on sys.path so
# tests can `import package_custom_build` etc. directly, the same way install.py puts REPO_ROOT
# on sys.path for its own internal imports.
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pytest


class RouterStub:
    """Plays llama-server's router over real HTTP, with a configurable cost model so a test can
    assert the fit recovers it -- exactly as the fake llama-bench in test_model_perf.py does.

    It SLEEPS for the duration it reports in `timings`, so a caller measuring wall clock and a
    caller reading the timings block see the same thing. That is what makes it usable for
    validate_model_perf.py, which compares predictions against measured wall clock; tests using it
    that way pick per-token costs small enough to keep the whole run to a couple of seconds.
    """

    def __init__(self, *, prefill_base=1.0, prefill_slope=0.0, decode_base=20.0, decode_slope=0.0,
                 decode_quadratic=0.0, omit_timings=False):
        self.requests: list[tuple[str, str, dict]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _read_body(self):
                length = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(length) or b"{}")

            def do_POST(self):
                body = self._read_body()
                outer.requests.append(("POST", self.path, body))
                if self.path == "/models/unload":
                    return self._respond({"success": True})
                prompt_n = max(1, len(body["messages"][0]["content"].split()))
                predicted_n = body.get("max_tokens", 32)
                prompt_ms = prompt_n * (prefill_base + prefill_slope * prompt_n / 2.0)
                depth = prompt_n + predicted_n / 2.0
                # decode_quadratic models a cost curve the linear-in-depth fit CANNOT see from
                # two shallow points -- the lever a test pulls to prove the validator goes red.
                decode_ms = predicted_n * (decode_base + decode_slope * depth + decode_quadratic * depth * depth)
                time.sleep((prompt_ms + decode_ms) / 1000.0)
                payload = {"choices": [{"message": {"content": "ok"}}]}
                if not omit_timings:
                    payload["timings"] = {
                        "prompt_n": prompt_n, "prompt_ms": prompt_ms,
                        "predicted_n": predicted_n, "predicted_ms": decode_ms,
                    }
                self._respond(payload)

            def do_DELETE(self):
                outer.requests.append(("DELETE", self.path, {}))
                self._respond({"success": True})

            def _respond(self, payload):
                raw = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self):
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def router_stub():
    """The RouterStub class itself, so a test can construct one with its own cost model."""
    return RouterStub

from aipotluck.installer.layout import Layout
from aipotluck.installer.platform_detect import HostProfile


@pytest.fixture
def linux_profile() -> HostProfile:
    return HostProfile(os_name="linux", arch="x64", backend="cpu")


@pytest.fixture
def fake_layout(tmp_path: Path) -> Layout:
    """A real Layout, rooted entirely under tmp_path -- every directory a test writes into is
    disposable and never touches the real per-OS install locations."""
    root = tmp_path / "install"
    lay = Layout(
        install_root=root,
        config_dir=root / "config",
        log_dir=root / "logs",
        state_dir=root / "state",
    )
    lay.config_dir.mkdir(parents=True, exist_ok=True)
    lay.log_dir.mkdir(parents=True, exist_ok=True)
    lay.state_dir.mkdir(parents=True, exist_ok=True)
    return lay
