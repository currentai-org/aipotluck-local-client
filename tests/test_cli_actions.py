"""aipotluck.installer.cli's run_login / run_logout / run_status.

Every OS/network/service-manager boundary is mocked: detect_host_profile (no real GPU probing),
newt_fetch (no real download), get_service_manager (no real systemd/launchd), and
urllib.request.urlopen (no real HTTP). Only runtime.json's actual read/write path is real,
rooted under tmp_path via `--install-dir`.
"""

from __future__ import annotations

import json
import logging
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aipotluck.installer import cli
from aipotluck.service import model_ops
from aipotluck.installer import model_sizing
from aipotluck.installer.model_sizing import SizingResult
from aipotluck.installer.platform_detect import HostProfile


@pytest.fixture(autouse=True)
def no_preflight_network(monkeypatch):
    """`pull` asks Hugging Face how big a model is before downloading it. Left alone that is a real
    HTTP call from every pull test -- slow, offline-hostile, and dependent on a repo that doesn't
    exist. Default every test to "no verdict" (the honest answer when the size can't be read) and
    let the tests that are about the gate say otherwise."""
    monkeypatch.setattr(model_ops.model_preflight, "check_fits_in_memory", lambda *a, **kw: None)


def make_args(install_dir: Path, **overrides) -> Namespace:
    defaults = dict(
        install_dir=install_dir,
        system=False,
        verbose=False,
        tunnel_id=None,
        tunnel_secret=None,
        tunnel_endpoint=None,
        credentials_file=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def write_runtime(install_dir: Path, config: dict) -> Path:
    config_dir = install_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "runtime.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _stable_profile(monkeypatch):
    """Every test in this file targets a deterministic linux profile -- nobody should be probing
    the real GPU/OS of whatever machine runs the suite."""
    monkeypatch.setattr(cli, "detect_host_profile", lambda: HostProfile(os_name="linux", arch="x64", backend="cpu"))


@pytest.fixture
def fake_service_manager(monkeypatch):
    mgr = MagicMock()
    monkeypatch.setattr(cli, "get_service_manager", lambda os_name: mgr)
    return mgr


@pytest.fixture
def fake_newt_fetch(monkeypatch, tmp_path):
    """Stubs the three newt_fetch calls run_login makes, in the shape it expects -- no network,
    no real binary written to disk (nothing downstream of run_login checks it exists)."""
    monkeypatch.setattr(cli.newt_fetch, "load_newt_manifest", lambda path: {"tag": "1.17.0", "assets": {}})
    monkeypatch.setattr(
        cli.newt_fetch, "resolve_newt_asset",
        lambda manifest, profile: ("linux-x64", {"file": "newt_linux_amd64"}),
    )
    fake_binary = tmp_path / "newt_linux_amd64"
    monkeypatch.setattr(cli.newt_fetch, "download_newt_binary", lambda manifest, entry, dest_dir: fake_binary)
    return fake_binary


class TestRunLogin:
    def test_with_all_three_flags_writes_pairing_and_logs_in(
        self, tmp_path, fake_service_manager, fake_newt_fetch, capsys
    ):
        install_dir = tmp_path / "install"
        runtime_path = write_runtime(install_dir, {"llama_cpp": {}, "service": {}, "logged_in": False})
        args = make_args(install_dir, tunnel_id="tid", tunnel_secret="tsecret", tunnel_endpoint="http://x")

        rc = cli.run_login(args)

        assert rc == 0
        saved = json.loads(runtime_path.read_text())
        assert saved["logged_in"] is True
        assert saved["tunnel"] == {
            "provider": "pangolin",
            "binary": str(fake_newt_fetch),
            "id": "tid",
            "secret": "tsecret",
            "endpoint": "http://x",
        }
        assert "Logged in." in capsys.readouterr().out

    def test_restarts_the_service(self, tmp_path, fake_service_manager, fake_newt_fetch):
        install_dir = tmp_path / "install"
        write_runtime(install_dir, {"llama_cpp": {}, "service": {}, "logged_in": False})
        args = make_args(install_dir, tunnel_id="tid", tunnel_secret="tsecret", tunnel_endpoint="http://x")

        cli.run_login(args)

        fake_service_manager.stop.assert_called_once()
        fake_service_manager.start.assert_called_once()

    def test_partial_tunnel_flags_is_rejected_before_touching_anything(self, tmp_path, fake_service_manager):
        # No runtime.json exists at this install_dir at all -- if run_login proceeded past its own
        # validation it would blow up on a missing file, which would also make this test fail, just
        # for the wrong reason. Asserting the return code (and that the service was never touched)
        # is what actually pins "reject early", not merely "doesn't crash".
        args = make_args(tmp_path / "install", tunnel_id="tid", tunnel_secret=None, tunnel_endpoint="http://x")

        rc = cli.run_login(args)

        assert rc == 1
        fake_service_manager.stop.assert_not_called()
        fake_service_manager.start.assert_not_called()

    def test_missing_runtime_json_exits_loudly(self, tmp_path, fake_service_manager, fake_newt_fetch):
        args = make_args(tmp_path / "never-installed", tunnel_id="tid", tunnel_secret="s", tunnel_endpoint="e")
        with pytest.raises(SystemExit):
            cli.run_login(args)

    def test_service_restart_failure_does_not_fail_the_login(self, tmp_path, monkeypatch, fake_newt_fetch):
        # A --no-service install has nothing registered to restart -- run_login must still report
        # success (the pairing itself was written correctly) rather than treating that as fatal.
        install_dir = tmp_path / "install"
        write_runtime(install_dir, {"llama_cpp": {}, "service": {}, "logged_in": False})

        def _raise_no_service_manager(os_name):
            raise RuntimeError("no service")

        monkeypatch.setattr(cli, "get_service_manager", _raise_no_service_manager)
        args = make_args(install_dir, tunnel_id="tid", tunnel_secret="s", tunnel_endpoint="e")

        rc = cli.run_login(args)

        assert rc == 0

    def test_prompts_for_credentials_json_when_no_flags_given(
        self, tmp_path, fake_service_manager, fake_newt_fetch, monkeypatch
    ):
        install_dir = tmp_path / "install"
        runtime_path = write_runtime(install_dir, {"llama_cpp": {}, "service": {}, "logged_in": False})
        monkeypatch.setattr(
            cli, "_prompt_credentials_json", lambda: ("prompted-id", "prompted-secret", "prompted-url")
        )
        args = make_args(install_dir)  # tunnel_* and credentials_file all left at their None default

        rc = cli.run_login(args)

        assert rc == 0
        saved = json.loads(runtime_path.read_text())
        assert saved["tunnel"]["id"] == "prompted-id"
        assert saved["tunnel"]["secret"] == "prompted-secret"
        assert saved["tunnel"]["endpoint"] == "prompted-url"

    def test_credentials_file_is_read_and_used(self, tmp_path, fake_service_manager, fake_newt_fetch):
        install_dir = tmp_path / "install"
        runtime_path = write_runtime(install_dir, {"llama_cpp": {}, "service": {}, "logged_in": False})
        creds_path = tmp_path / "creds.json"
        creds_path.write_text(
            json.dumps({"tunnelId": "file-id", "tunnelSecret": "file-secret", "tunnelEndpoint": "file-url"}),
            encoding="utf-8",
        )
        args = make_args(install_dir, credentials_file=creds_path)

        rc = cli.run_login(args)

        assert rc == 0
        saved = json.loads(runtime_path.read_text())
        assert saved["tunnel"]["id"] == "file-id"
        assert saved["tunnel"]["secret"] == "file-secret"
        assert saved["tunnel"]["endpoint"] == "file-url"

    def test_credentials_file_with_invalid_json_is_rejected(self, tmp_path, fake_service_manager, caplog):
        install_dir = tmp_path / "install"
        write_runtime(install_dir, {"llama_cpp": {}, "service": {}, "logged_in": False})
        creds_path = tmp_path / "creds.json"
        creds_path.write_text("not json", encoding="utf-8")
        args = make_args(install_dir, credentials_file=creds_path)

        with caplog.at_level(logging.ERROR, logger="aipotluck.cli"):
            rc = cli.run_login(args)

        assert rc == 1
        assert any("not valid JSON" in rec.message for rec in caplog.records)

    def test_missing_credentials_file_is_rejected(self, tmp_path, fake_service_manager):
        install_dir = tmp_path / "install"
        write_runtime(install_dir, {"llama_cpp": {}, "service": {}, "logged_in": False})
        args = make_args(install_dir, credentials_file=tmp_path / "does-not-exist.json")

        rc = cli.run_login(args)

        assert rc == 1

    def test_credentials_file_and_tunnel_flags_together_is_rejected(self, tmp_path, fake_service_manager):
        args = make_args(
            tmp_path / "install", credentials_file=tmp_path / "creds.json",
            tunnel_id="tid", tunnel_secret="s", tunnel_endpoint="e",
        )

        rc = cli.run_login(args)

        assert rc == 1
        fake_service_manager.stop.assert_not_called()


class TestParseCredentialsJson:
    def test_parses_valid_json(self):
        raw = json.dumps({"tunnelId": "i", "tunnelSecret": "s", "tunnelEndpoint": "e"})
        assert cli._parse_credentials_json(raw) == ("i", "s", "e")

    def test_rejects_invalid_json(self):
        with pytest.raises(cli.CredentialsError, match="not valid JSON"):
            cli._parse_credentials_json("{not json")

    def test_rejects_a_json_array(self):
        with pytest.raises(cli.CredentialsError, match="JSON object"):
            cli._parse_credentials_json("[1, 2, 3]")

    def test_rejects_missing_key(self):
        raw = json.dumps({"tunnelId": "i", "tunnelSecret": "s"})  # no tunnelEndpoint
        with pytest.raises(cli.CredentialsError, match="tunnelEndpoint"):
            cli._parse_credentials_json(raw)

    def test_rejects_empty_string_value(self):
        raw = json.dumps({"tunnelId": "", "tunnelSecret": "s", "tunnelEndpoint": "e"})
        with pytest.raises(cli.CredentialsError, match="tunnelId"):
            cli._parse_credentials_json(raw)

    def test_rejects_non_string_value(self):
        raw = json.dumps({"tunnelId": 12345, "tunnelSecret": "s", "tunnelEndpoint": "e"})
        with pytest.raises(cli.CredentialsError, match="tunnelId"):
            cli._parse_credentials_json(raw)


class TestPromptCredentialsJson:
    def test_returns_parsed_credentials_on_first_valid_paste(self, monkeypatch):
        raw = json.dumps({"tunnelId": "i", "tunnelSecret": "s", "tunnelEndpoint": "e"})
        monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt_text: raw)

        assert cli._prompt_credentials_json() == ("i", "s", "e")

    def test_retries_on_empty_paste(self, monkeypatch, capsys):
        raw = json.dumps({"tunnelId": "i", "tunnelSecret": "s", "tunnelEndpoint": "e"})
        answers = iter(["", "   ", raw])
        monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt_text: next(answers))

        result = cli._prompt_credentials_json()

        assert result == ("i", "s", "e")
        assert capsys.readouterr().out.count("(required, try again)") == 2

    def test_retries_on_invalid_json(self, monkeypatch, capsys):
        valid = json.dumps({"tunnelId": "i", "tunnelSecret": "s", "tunnelEndpoint": "e"})
        answers = iter(["not json at all", valid])
        monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt_text: next(answers))

        result = cli._prompt_credentials_json()

        assert result == ("i", "s", "e")
        assert "Try again" in capsys.readouterr().out

    def test_never_echoes_input(self, monkeypatch):
        # The whole point of using getpass here is that the secret embedded in the pasted JSON is
        # never echoed -- assert the prompt path goes through getpass.getpass, not the plain
        # echoing `input()` builtin.
        calls = []
        raw = json.dumps({"tunnelId": "i", "tunnelSecret": "s", "tunnelEndpoint": "e"})
        monkeypatch.setattr(cli.getpass, "getpass", lambda prompt_text: calls.append(prompt_text) or raw)

        def _fail_if_called(*_args):
            raise AssertionError("must not use input() -- it would echo the secret")

        monkeypatch.setattr(cli, "input", _fail_if_called, raising=False)

        cli._prompt_credentials_json()

        assert len(calls) == 1


class TestRunLogout:
    def test_removes_tunnel_and_flips_logged_in(self, tmp_path, fake_service_manager):
        install_dir = tmp_path / "install"
        runtime_path = write_runtime(
            install_dir,
            {"llama_cpp": {}, "service": {}, "logged_in": True, "tunnel": {"id": "x", "secret": "y", "endpoint": "z"}},
        )
        args = make_args(install_dir)

        rc = cli.run_logout(args)

        assert rc == 0
        saved = json.loads(runtime_path.read_text())
        assert saved["logged_in"] is False
        assert "tunnel" not in saved
        fake_service_manager.stop.assert_called_once()
        fake_service_manager.start.assert_called_once()

    def test_already_logged_out_is_a_noop(self, tmp_path, fake_service_manager, caplog):
        install_dir = tmp_path / "install"
        write_runtime(install_dir, {"llama_cpp": {}, "service": {}, "logged_in": False})
        args = make_args(install_dir)

        with caplog.at_level(logging.INFO, logger="aipotluck.cli"):
            rc = cli.run_logout(args)

        assert rc == 0
        assert any("Already logged out" in rec.message for rec in caplog.records)
        fake_service_manager.stop.assert_not_called()
        fake_service_manager.start.assert_not_called()

    def test_missing_runtime_json_exits_loudly(self, tmp_path, fake_service_manager):
        args = make_args(tmp_path / "never-installed")
        with pytest.raises(SystemExit):
            cli.run_logout(args)


class _FakeHttpResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return self._body


class TestRunStatus:
    def test_reports_logged_in_state_and_process_info(self, monkeypatch, capsys):
        payload = {
            "logged_in": True,
            "llama_server": {"pid": 123, "running": True},
            "tunnel": {"pid": 456, "running": True},
        }
        monkeypatch.setattr(cli.urllib.request, "urlopen", lambda url, timeout: _FakeHttpResponse(payload))

        rc = cli.run_status(None)

        out = capsys.readouterr().out
        assert rc == 0
        assert "Login:        logged in" in out
        assert "123" in out
        assert "456" in out

    def test_reports_logged_out_state_without_error_noise(self, monkeypatch, capsys):
        payload = {"logged_in": False, "llama_server": None, "tunnel": None}
        monkeypatch.setattr(cli.urllib.request, "urlopen", lambda url, timeout: _FakeHttpResponse(payload))

        rc = cli.run_status(None)

        out = capsys.readouterr().out
        assert rc == 0
        assert "Login:        logged out" in out
        assert "stopped (logged out)" in out

    def test_unreachable_service_returns_error(self, monkeypatch, caplog):
        import urllib.error

        def raise_it(url, timeout):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(cli.urllib.request, "urlopen", raise_it)

        with caplog.at_level(logging.ERROR, logger="aipotluck.cli"):
            rc = cli.run_status(None)

        assert rc == 1
        assert any("Could not reach" in rec.message for rec in caplog.records)

    def test_running_but_disconnected_tunnel_gets_an_explicit_warning(self, monkeypatch, capsys):
        # The exact discrepancy reported live: newt "running: true" for 20+ hours while its own
        # retry loop never once actually connected -- a plain dict print alone buries this.
        payload = {
            "logged_in": True,
            "llama_server": {"pid": 1, "running": True},
            "tunnel": {"pid": 456, "running": True, "tunnel_connected": False},
        }
        monkeypatch.setattr(cli.urllib.request, "urlopen", lambda url, timeout: _FakeHttpResponse(payload))

        cli.run_status(None)

        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "newt.log" in out

    def test_running_and_connected_tunnel_has_no_warning(self, monkeypatch, capsys):
        payload = {
            "logged_in": True,
            "llama_server": {"pid": 1, "running": True},
            "tunnel": {"pid": 456, "running": True, "tunnel_connected": True},
        }
        monkeypatch.setattr(cli.urllib.request, "urlopen", lambda url, timeout: _FakeHttpResponse(payload))

        cli.run_status(None)

        out = capsys.readouterr().out
        assert "WARNING" not in out

    def test_unknown_tunnel_connected_state_has_no_warning(self, monkeypatch, capsys):
        # tunnel_connected: None (no log signal yet) must not be treated as "definitely down".
        payload = {
            "logged_in": True,
            "llama_server": {"pid": 1, "running": True},
            "tunnel": {"pid": 456, "running": True, "tunnel_connected": None},
        }
        monkeypatch.setattr(cli.urllib.request, "urlopen", lambda url, timeout: _FakeHttpResponse(payload))

        cli.run_status(None)

        out = capsys.readouterr().out
        assert "WARNING" not in out

    def test_runtime_params_are_shown(self, monkeypatch, capsys):
        payload = {
            "logged_in": True,
            "llama_server": {"pid": 1, "running": True},
            "tunnel": None,
            "runtime_params": {
                "gpu_layers": "auto", "models_max": 1,
                "models": {"org/repo:Q4_K_M": {"ctx_size": "32768", "parallel": "1", "tuning": {}}},
            },
        }
        monkeypatch.setattr(cli.urllib.request, "urlopen", lambda url, timeout: _FakeHttpResponse(payload))

        cli.run_status(None)

        out = capsys.readouterr().out
        assert "Router params: gpu_layers=auto, models_max=1" in out
        assert "Sized models: {'org/repo:Q4_K_M': {'ctx_size': '32768', 'parallel': '1'}}" in out

    def test_sized_models_is_exactly_one_line_per_the_tunnel_status_convention(self, monkeypatch, capsys):
        # Same terse "one line, raw dict" shape as the llama-server/Tunnel lines above it --
        # multiple models must never spill onto their own separate lines.
        payload = {
            "logged_in": True,
            "llama_server": {"pid": 1, "running": True},
            "tunnel": None,
            "runtime_params": {
                "gpu_layers": "auto", "models_max": 1,
                "models": {
                    "org/a:Q4_K_M": {"ctx_size": "8192", "parallel": "1", "tuning": {}},
                    "org/b:Q8_0": {"ctx_size": "4096", "parallel": "1", "tuning": {}},
                },
            },
        }
        monkeypatch.setattr(cli.urllib.request, "urlopen", lambda url, timeout: _FakeHttpResponse(payload))

        cli.run_status(None)

        out = capsys.readouterr().out
        sized_lines = [line for line in out.splitlines() if line.startswith("Sized models")]
        assert len(sized_lines) == 1
        assert "org/a:Q4_K_M" in sized_lines[0] and "org/b:Q8_0" in sized_lines[0]

    def test_tuning_reasons_are_omitted_from_status_output(self, monkeypatch, capsys):
        # The CLI's `status` is deliberately terse (one line, like the Tunnel/llama-server lines
        # above it) -- the per-field "why" reasoning is real and useful, but belongs in
        # GET /capabilities, not in a quick-glance status line.
        payload = {
            "logged_in": True,
            "llama_server": {"pid": 1, "running": True},
            "tunnel": None,
            "runtime_params": {
                "gpu_layers": None, "models_max": 1,
                "models": {
                    "org/repo:Q4_K_M": {
                        "ctx_size": "32768", "parallel": "1",
                        "tuning": {
                            "ctx_size": "capped by available memory",
                            "parallel": "reduced from the default to maximize single-request context",
                        },
                    }
                },
            },
        }
        monkeypatch.setattr(cli.urllib.request, "urlopen", lambda url, timeout: _FakeHttpResponse(payload))

        cli.run_status(None)

        out = capsys.readouterr().out
        assert "capped by available memory" not in out
        assert "reduced from the default to maximize single-request context" not in out
        assert "tuning" not in out

    def test_runtime_params_absent_prints_nothing(self, monkeypatch, capsys):
        # Old service builds predating runtime_params -- must not crash on a missing key.
        payload = {"logged_in": True, "llama_server": {"pid": 1, "running": True}, "tunnel": None}
        monkeypatch.setattr(cli.urllib.request, "urlopen", lambda url, timeout: _FakeHttpResponse(payload))

        rc = cli.run_status(None)

        assert rc == 0
        out = capsys.readouterr().out
        assert "Runtime params" not in out

    def test_logged_in_but_llama_supervisor_errored_is_surfaced(self, monkeypatch, capsys):
        payload = {
            "logged_in": True,
            "llama_server": {"error": "supervisor not started"},
            "tunnel": {"pid": 1, "running": True},
        }
        monkeypatch.setattr(cli.urllib.request, "urlopen", lambda url, timeout: _FakeHttpResponse(payload))

        cli.run_status(None)

        out = capsys.readouterr().out
        assert "not running (supervisor not started)" in out


def make_pull_args(install_dir: Path, model: str = "org/repo:Q4_K_M", timeout=None, **overrides) -> Namespace:
    defaults = dict(
        install_dir=install_dir, system=False, verbose=False, model=model, timeout=timeout,
        force=False, skip_benchmark=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


_FAKE_SIZING = SizingResult(
    ctx_size=16384, parallel=1, cache_type_k="q8_0", cache_type_v="q8_0",
    tuning={"ctx_size": "test reason"},
)


class TestReloadRouterModels:
    """Real HTTP throughout -- a small stand-in HTTP server plays the router, so this proves the
    actual request cli.py sends (path/query), not just that some call happened."""

    def test_hits_models_reload_on_the_configured_host_and_port(self):
        import socket
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        seen_paths = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                seen_paths.append(self.path)
                self.send_response(200)
                self.end_headers()

            def log_message(self, *a):
                pass

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        server = HTTPServer(("127.0.0.1", port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = model_ops.reload_router({"host": "127.0.0.1", "port": port})
        finally:
            server.shutdown()
            thread.join(timeout=5)

        assert result is True
        assert seen_paths == ["/models?reload=1"]

    def test_returns_false_rather_than_raising_when_nothing_is_listening(self):
        # An unbound ephemeral port -- guaranteed connection-refused, the realistic "not logged in
        # / service not running" case.
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        assert model_ops.reload_router({"host": "127.0.0.1", "port": port}) is False

    def test_defaults_to_localhost_8080_when_unset(self, monkeypatch):
        # Must not raise with an empty llama_cfg -- falls back to llama-server's own conventional
        # default host/port rather than crashing on a missing key. The real default is the point
        # here, so lift conftest's port guard -- urlopen is faked below, so nothing is reached.
        monkeypatch.undo()
        seen_urls = []

        def fake_urlopen(url, timeout):
            seen_urls.append(url)
            return _FakeHttpResponse({})

        monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
        assert model_ops.reload_router({}) is True
        assert seen_urls == ["http://127.0.0.1:8080/models?reload=1"]


class TestOversizedModelGate:
    """Refusing a download whose weights are bigger than the machine.

    This is the only check that runs before the download, so it is also the only one that can save
    the user the download. It is deliberately advisory: it asks, it does not decide, and an absent
    verdict always proceeds.
    """

    def _setup(self, tmp_path, monkeypatch, verdict, *, tty=True):
        install_dir = tmp_path / "install"
        write_runtime(install_dir, {
            "llama_cpp": {"server_binary": "/fake/llama-server", "presets_path": "/cfg/presets.ini"},
            "service": {},
        })
        pulls = []
        monkeypatch.setattr(model_ops, "pull_model", lambda *a, **kw: pulls.append(1))
        monkeypatch.setattr(model_ops.model_sizing, "ensure_preset", lambda *a, **kw: _FAKE_SIZING)
        monkeypatch.setattr(model_ops, "reload_router", lambda llama_cfg: True)
        monkeypatch.setattr(model_ops, "measure", lambda *a, **kw: (None, None, None))
        monkeypatch.setattr(model_ops.model_preflight, "check_fits_in_memory", lambda *a, **kw: verdict)
        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: tty, raising=False)
        return install_dir, pulls

    @staticmethod
    def _verdict(fits):
        return model_ops.model_preflight.PreflightVerdict(
            fits=fits, model_bytes=8 * 1024**3, total_memory_bytes=4 * 1024**3,
            detail="org/repo:Q4_K_M is 8.0GB of weights, and this device has 4.0GB of memory in total.",
        )

    def test_declining_the_prompt_stops_before_downloading(self, tmp_path, monkeypatch, capsys):
        install_dir, pulls = self._setup(tmp_path, monkeypatch, self._verdict(False))
        monkeypatch.setattr("builtins.input", lambda *a: "n")

        rc = cli.run_pull_model(make_pull_args(install_dir))

        assert rc == 1
        assert pulls == [], "the whole point is not spending the download"
        assert "WARNING" in capsys.readouterr().out

    def test_accepting_the_prompt_downloads_anyway(self, tmp_path, monkeypatch):
        install_dir, pulls = self._setup(tmp_path, monkeypatch, self._verdict(False))
        monkeypatch.setattr("builtins.input", lambda *a: "y")
        assert cli.run_pull_model(make_pull_args(install_dir)) == 0
        assert pulls == [1]

    def test_the_default_answer_is_no(self, tmp_path, monkeypatch):
        """Empty input must not be read as consent -- this prompt spends gigabytes."""
        install_dir, pulls = self._setup(tmp_path, monkeypatch, self._verdict(False))
        monkeypatch.setattr("builtins.input", lambda *a: "")
        assert cli.run_pull_model(make_pull_args(install_dir)) == 1
        assert pulls == []

    def test_a_model_that_fits_is_never_mentioned(self, tmp_path, monkeypatch, capsys):
        install_dir, pulls = self._setup(tmp_path, monkeypatch, self._verdict(True))
        monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("must not prompt"))
        assert cli.run_pull_model(make_pull_args(install_dir)) == 0
        assert pulls == [1] and "WARNING" not in capsys.readouterr().out

    def test_no_verdict_proceeds_without_a_prompt(self, tmp_path, monkeypatch):
        """Offline, rate-limited or a private repo. Absence of evidence has to let the pull through
        to the layers that measure for real, or this gate becomes an outage."""
        install_dir, pulls = self._setup(tmp_path, monkeypatch, None)
        monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("must not prompt"))
        assert cli.run_pull_model(make_pull_args(install_dir)) == 0
        assert pulls == [1]

    def test_force_skips_the_check_entirely(self, tmp_path, monkeypatch):
        lookups = []
        install_dir, pulls = self._setup(tmp_path, monkeypatch, self._verdict(False))
        monkeypatch.setattr(model_ops.model_preflight, "check_fits_in_memory",
                            lambda *a, **kw: lookups.append(1) or self._verdict(False))
        assert cli.run_pull_model(make_pull_args(install_dir, force=True)) == 0
        assert pulls == [1]
        assert lookups == [], "--force means don't even ask Hugging Face"

    def test_without_a_terminal_it_refuses_instead_of_prompting(self, tmp_path, monkeypatch, capsys):
        """A piped or scripted pull has nobody to answer, so blocking on input would hang and
        assuming yes would spend the download. Stop, and name the flag that overrides."""
        install_dir, pulls = self._setup(tmp_path, monkeypatch, self._verdict(False), tty=False)
        monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("must not prompt"))

        assert cli.run_pull_model(make_pull_args(install_dir)) == 1
        assert pulls == []
        assert "--force" in capsys.readouterr().out


class TestRunPullModel:
    """Router mode (CUR-1965's follow-up): there's no "active model" to write into runtime.json
    anymore -- pull downloads a model, sizes it (model_sizing.ensure_preset persists the
    --models-preset INI section + its tuning-reasons sidecar on its own, see
    test_model_sizing.py/test_model_presets.py for that), then best-effort asks a running router
    to reload rather than restarting the whole service."""

    def test_success_pulls_sizes_and_reloads_the_router(self, tmp_path, monkeypatch):
        install_dir = tmp_path / "install"
        write_runtime(
            install_dir,
            {"llama_cpp": {"server_binary": "/fake/llama-server", "presets_path": "/cfg/presets.ini"}, "service": {}},
        )
        pull_calls = []
        monkeypatch.setattr(model_ops, "pull_model", lambda *a, **kw: pull_calls.append((a, kw)))
        ensure_preset_calls = []
        monkeypatch.setattr(model_ops.model_sizing, "ensure_preset", lambda *a, **kw: ensure_preset_calls.append((a, kw)) or _FAKE_SIZING)
        reload_calls = []
        monkeypatch.setattr(model_ops, "reload_router", lambda llama_cfg: reload_calls.append(1) or True)
        args = make_pull_args(install_dir, model="new/model:Q8_0")

        rc = cli.run_pull_model(args)

        assert rc == 0
        assert len(pull_calls) == 1
        assert pull_calls[0][0] == (Path("/fake/llama-server"), "new/model:Q8_0")
        assert len(ensure_preset_calls) == 1
        _, kwargs = ensure_preset_calls[0]
        assert kwargs["model_hf"] == "new/model:Q8_0"
        assert kwargs["force"] is True
        assert reload_calls == [1]

    def test_missing_llama_cpp_install_is_rejected_before_pulling(self, tmp_path, monkeypatch):
        install_dir = tmp_path / "install"
        write_runtime(install_dir, {"service": {}, "logged_in": False})  # no llama_cpp section at all
        pull_calls = []
        monkeypatch.setattr(model_ops, "pull_model", lambda *a, **kw: pull_calls.append((a, kw)))
        args = make_pull_args(install_dir)

        rc = cli.run_pull_model(args)

        assert rc == 1
        assert pull_calls == []

    def test_pull_failure_skips_sizing_entirely(self, tmp_path, monkeypatch):
        install_dir = tmp_path / "install"
        write_runtime(
            install_dir, {"llama_cpp": {"server_binary": "/fake/llama-server"}, "service": {}}
        )

        def raise_pull_error(*a, **kw):
            raise cli.ModelPullError("bad repo/quant")

        monkeypatch.setattr(model_ops, "pull_model", raise_pull_error)
        ensure_preset_calls = []
        monkeypatch.setattr(model_ops.model_sizing, "ensure_preset", lambda *a, **kw: ensure_preset_calls.append(1))
        args = make_pull_args(install_dir, model="bad/repo")

        rc = cli.run_pull_model(args)

        assert rc == 1
        assert ensure_preset_calls == []

    def test_default_timeout_used_when_not_given(self, tmp_path, monkeypatch):
        install_dir = tmp_path / "install"
        write_runtime(install_dir, {"llama_cpp": {"server_binary": "/fake/llama-server"}, "service": {}})
        seen_kwargs = {}
        monkeypatch.setattr(model_ops, "pull_model", lambda *a, **kw: seen_kwargs.update(kw))
        monkeypatch.setattr(model_ops.model_sizing, "ensure_preset", lambda *a, **kw: _FAKE_SIZING)
        args = make_pull_args(install_dir, timeout=None)

        cli.run_pull_model(args)

        assert seen_kwargs["timeout"] == cli.DEFAULT_TIMEOUT_SECONDS

    def test_custom_timeout_passed_through(self, tmp_path, monkeypatch):
        install_dir = tmp_path / "install"
        write_runtime(install_dir, {"llama_cpp": {"server_binary": "/fake/llama-server"}, "service": {}})
        seen_kwargs = {}
        monkeypatch.setattr(model_ops, "pull_model", lambda *a, **kw: seen_kwargs.update(kw))
        monkeypatch.setattr(model_ops.model_sizing, "ensure_preset", lambda *a, **kw: _FAKE_SIZING)
        args = make_pull_args(install_dir, timeout=45.0)

        cli.run_pull_model(args)

        assert seen_kwargs["timeout"] == 45.0

    def test_gpu_layers_forwarded_to_both_pull_and_sizing(self, tmp_path, monkeypatch):
        install_dir = tmp_path / "install"
        write_runtime(
            install_dir,
            {
                "llama_cpp": {
                    "server_binary": "/fake/llama-server", "gpu_layers": "all", "presets_path": "/cfg/presets.ini",
                },
                "service": {},
            },
        )
        pull_kwargs = {}
        monkeypatch.setattr(model_ops, "pull_model", lambda *a, **kw: pull_kwargs.update(kw))
        ensure_preset_kwargs = {}
        monkeypatch.setattr(
            model_ops.model_sizing, "ensure_preset",
            lambda *a, **kw: ensure_preset_kwargs.update(kw) or _FAKE_SIZING,
        )
        args = make_pull_args(install_dir)

        cli.run_pull_model(args)

        assert pull_kwargs["gpu_layers"] == "all"
        assert ensure_preset_kwargs["gpu_layers"] == "all"

    def test_no_presets_path_skips_sizing_but_still_succeeds(self, tmp_path, monkeypatch, caplog):
        install_dir = tmp_path / "install"
        write_runtime(install_dir, {"llama_cpp": {"server_binary": "/fake/llama-server"}, "service": {}})
        monkeypatch.setattr(model_ops, "pull_model", lambda *a, **kw: None)
        ensure_preset_calls = []
        monkeypatch.setattr(model_ops.model_sizing, "ensure_preset", lambda *a, **kw: ensure_preset_calls.append(1))
        args = make_pull_args(install_dir)

        with caplog.at_level(logging.WARNING, logger="aipotluck.cli"):
            rc = cli.run_pull_model(args)

        assert rc == 0
        assert ensure_preset_calls == []
        assert any("No presets_path configured" in rec.message for rec in caplog.records)

    def test_sizing_failure_does_not_fail_the_pull(self, tmp_path, monkeypatch, caplog):
        # A nice-to-have auto-tune layer failing (e.g. the probe times out, or this platform can't
        # measure memory) must not block the actual model switch -- the pull already succeeded.
        install_dir = tmp_path / "install"
        write_runtime(
            install_dir,
            {"llama_cpp": {"server_binary": "/fake/llama-server", "presets_path": "/cfg/presets.ini"}, "service": {}},
        )
        monkeypatch.setattr(model_ops, "pull_model", lambda *a, **kw: None)

        def raise_sizing_error(*a, **kw):
            raise cli.ModelSizingError("probe timed out")

        monkeypatch.setattr(model_ops.model_sizing, "ensure_preset", raise_sizing_error)
        args = make_pull_args(install_dir, model="new/model:Q8_0")

        with caplog.at_level(logging.WARNING, logger="aipotluck.cli"):
            rc = cli.run_pull_model(args)

        assert rc == 0  # the pull itself still succeeds
        assert any("Automatic runtime sizing failed" in rec.message for rec in caplog.records)

    def test_reload_failure_still_reports_success_with_a_different_message(self, tmp_path, monkeypatch, capsys):
        install_dir = tmp_path / "install"
        write_runtime(
            install_dir,
            {"llama_cpp": {"server_binary": "/fake/llama-server", "presets_path": "/cfg/presets.ini"}, "service": {}},
        )
        monkeypatch.setattr(model_ops, "pull_model", lambda *a, **kw: None)
        monkeypatch.setattr(model_ops.model_sizing, "ensure_preset", lambda *a, **kw: _FAKE_SIZING)
        monkeypatch.setattr(model_ops, "reload_router", lambda llama_cfg: False)
        args = make_pull_args(install_dir)

        rc = cli.run_pull_model(args)

        assert rc == 0
        out = capsys.readouterr().out
        assert "next time it starts" in out


class TestQuarantineSurfacing:
    """A held-back model is still on disk and still pickable in the chat UI -- the router
    discovers the HF cache for itself. `list` therefore has to say so plainly, because the
    alternative reading is that the service already dealt with it, leaving the user waiting on a
    turn that keeps failing.
    """

    def _setup(self, tmp_path, monkeypatch, cached=("org/repo:Q4_K_M",)):
        install_dir = tmp_path / "install"
        presets_path = tmp_path / "presets.ini"
        cli.model_presets.write_preset(presets_path, "org/repo:Q4_K_M", {"ctx-size": "4096"})
        write_runtime(install_dir, {
            "llama_cpp": {"server_binary": "/fake/llama-server", "presets_path": str(presets_path)},
            "service": {},
        })
        monkeypatch.setattr(model_ops, "list_cached_models", lambda binary: list(cached))
        return install_dir

    def _config_dir(self, install_dir):
        return install_dir / "config"

    def test_a_held_back_model_is_marked_and_explained(self, tmp_path, monkeypatch, capsys):
        install_dir = self._setup(tmp_path, monkeypatch)
        model_ops.model_health.quarantine(
            self._config_dir(install_dir), "org/repo:Q4_K_M",
            "ran out of memory 3 times without completing a turn in between",
        )

        assert cli.run_list_models(make_pull_args(install_dir)) == 0

        out = capsys.readouterr().out
        assert "HELD BACK" in out
        assert "ran out of memory 3 times" in out, "the reason has to reach the user, not just a flag"
        assert "still installed and can still be picked" in out
        assert "remove <model>" in out

    def test_a_held_back_model_is_not_told_to_run_benchmark(self, tmp_path, monkeypatch, capsys):
        """Benchmarking is usually what held it back -- a model that cannot be loaded is
        discovered by exactly that command -- so the suggestion would send the user round the
        same loop and watch it fail again."""
        install_dir = self._setup(tmp_path, monkeypatch)
        cli.model_health.quarantine(
            self._config_dir(install_dir), "org/repo:Q4_K_M", "it ran out of memory while loading",
        )

        cli.run_list_models(make_pull_args(install_dir))

        out = capsys.readouterr().out
        assert "HELD BACK" in out
        assert "run `benchmark`" not in out

    def test_an_unmeasured_model_that_is_fine_is_still_told_to_run_benchmark(
        self, tmp_path, monkeypatch, capsys
    ):
        """The other half of the pair -- the advice must not disappear for everyone."""
        install_dir = self._setup(tmp_path, monkeypatch)
        cli.run_list_models(make_pull_args(install_dir))
        assert "run `benchmark`" in capsys.readouterr().out

    def test_nothing_is_said_when_nothing_is_held_back(self, tmp_path, monkeypatch, capsys):
        install_dir = self._setup(tmp_path, monkeypatch)
        cli.run_list_models(make_pull_args(install_dir))
        out = capsys.readouterr().out
        assert "HELD BACK" not in out and "Held back" not in out


def make_remove_args(install_dir: Path, model="org/repo:Q4_K_M", yes=False) -> Namespace:
    return Namespace(install_dir=install_dir, system=False, verbose=False, model=model, yes=yes)


class TestRunRemoveModel:
    """The user-confirmed half of a quarantine. The service never deletes a model on its own --
    those are gigabytes the user chose to download -- so the decision lands here."""

    def _setup(self, tmp_path, monkeypatch, *, cached=("org/repo:Q4_K_M",), deletes_ok=True):
        install_dir = tmp_path / "install"
        presets_path = tmp_path / "presets.ini"
        cli.model_presets.write_preset(presets_path, "org/repo:Q4_K_M", {"ctx-size": "4096"})
        write_runtime(install_dir, {
            "llama_cpp": {"server_binary": "/fake/llama-server", "presets_path": str(presets_path)},
            "service": {},
        })
        monkeypatch.setattr(model_ops, "list_cached_models", lambda binary: list(cached))
        deletes = []

        def delete(cfg, mid):
            deletes.append(mid)
            if not deletes_ok:
                raise model_ops.ModelOpError(
                    f"Could not remove {mid}: model name={mid} is not removable (not from cache)",
                    code="delete_failed", status=502,
                )

        monkeypatch.setattr(model_ops, "delete_cached_model", delete)
        monkeypatch.setattr(model_ops, "reload_router", lambda cfg: True)
        return install_dir, deletes

    def test_a_removed_models_preset_goes_with_it(self, tmp_path, monkeypatch):
        """Otherwise the router keeps a section for a model it can no longer serve, and `status`
        keeps explaining the sizing of something that isn't there."""
        install_dir, _ = self._setup(tmp_path, monkeypatch)
        assert cli.run_remove_model(make_remove_args(install_dir, yes=True)) == 0
        assert cli.model_presets.read_all(tmp_path / "presets.ini") == {}

    def test_a_failed_delete_keeps_the_preset(self, tmp_path, monkeypatch):
        install_dir, _ = self._setup(tmp_path, monkeypatch, deletes_ok=False)
        assert cli.run_remove_model(make_remove_args(install_dir, yes=True)) == 1
        assert "org/repo:Q4_K_M" in cli.model_presets.read_all(tmp_path / "presets.ini")

    def test_yes_removes_without_prompting(self, tmp_path, monkeypatch):
        install_dir, deletes = self._setup(tmp_path, monkeypatch)
        monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("must not prompt with --yes"))
        assert cli.run_remove_model(make_remove_args(install_dir, yes=True)) == 0
        assert deletes == ["org/repo:Q4_K_M"]

    def test_declining_the_prompt_keeps_the_model(self, tmp_path, monkeypatch, capsys):
        install_dir, deletes = self._setup(tmp_path, monkeypatch)
        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr("builtins.input", lambda *a: "n")
        assert cli.run_remove_model(make_remove_args(install_dir)) == 1
        assert deletes == [] and "Left alone" in capsys.readouterr().out

    def test_the_default_answer_is_no(self, tmp_path, monkeypatch):
        """Empty input must not delete several gigabytes."""
        install_dir, deletes = self._setup(tmp_path, monkeypatch)
        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr("builtins.input", lambda *a: "")
        assert cli.run_remove_model(make_remove_args(install_dir)) == 1
        assert deletes == []

    def test_without_a_terminal_it_requires_the_flag(self, tmp_path, monkeypatch):
        install_dir, deletes = self._setup(tmp_path, monkeypatch)
        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False, raising=False)
        monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("must not prompt"))
        assert cli.run_remove_model(make_remove_args(install_dir)) == 1
        assert deletes == []

    def test_an_unknown_model_is_rejected_before_anything_is_deleted(self, tmp_path, monkeypatch):
        install_dir, deletes = self._setup(tmp_path, monkeypatch)
        rc = cli.run_remove_model(make_remove_args(install_dir, model="org/nope:Q4_K_M", yes=True))
        assert rc == 1 and deletes == []

    def test_the_quarantine_reason_is_shown_before_deleting(self, tmp_path, monkeypatch, capsys):
        install_dir, _ = self._setup(tmp_path, monkeypatch)
        model_ops.model_health.quarantine(
            install_dir / "config", "org/repo:Q4_K_M", "ran out of memory 3 times"
        )
        cli.run_remove_model(make_remove_args(install_dir, yes=True))
        assert "ran out of memory 3 times" in capsys.readouterr().out

    def test_removal_clears_the_health_record(self, tmp_path, monkeypatch):
        """A later re-pull of the same id must start clean rather than inherit a verdict reached
        against a sizing that no longer exists."""
        install_dir, _ = self._setup(tmp_path, monkeypatch)
        config_dir = install_dir / "config"
        model_ops.model_health.quarantine(config_dir, "org/repo:Q4_K_M", "three strikes")
        cli.run_remove_model(make_remove_args(install_dir, yes=True))
        assert model_ops.model_health.read_record(config_dir, "org/repo:Q4_K_M") is None

    def test_a_failed_delete_keeps_the_records_and_passes_on_llama_cpps_reason(
        self, tmp_path, monkeypatch, capsys
    ):
        """The files are still there, so forgetting the model would leave it installed, unexplained
        and un-held-back. And the reason is llama.cpp's own -- not a guess that the service wasn't
        reachable, which is what this used to say even when the router had answered."""
        install_dir, _ = self._setup(tmp_path, monkeypatch, deletes_ok=False)
        config_dir = install_dir / "config"
        model_ops.model_health.quarantine(config_dir, "org/repo:Q4_K_M", "three strikes")

        rc = cli.run_remove_model(make_remove_args(install_dir, yes=True))

        assert rc == 1
        assert model_ops.model_health.read_record(config_dir, "org/repo:Q4_K_M") is not None
        out = capsys.readouterr()
        assert "not removable (not from cache)" in out.out + out.err
        assert "wasn't reachable" not in out.out + out.err


class TestRunListModels:
    def test_lists_cached_models_and_flags_the_unsized_one(self, tmp_path, monkeypatch, capsys):
        install_dir = tmp_path / "install"
        presets_path = tmp_path / "presets.ini"
        cli.model_presets.write_preset(presets_path, "org/repo:Q4_K_M", {"ctx-size": "4096"})
        write_runtime(
            install_dir,
            {
                "llama_cpp": {"server_binary": "/fake/llama-server", "presets_path": str(presets_path)},
                "service": {},
            },
        )
        monkeypatch.setattr(model_ops, "list_cached_models", lambda binary: ["org/repo:Q4_K_M", "org/other:Q8_0"])
        args = make_pull_args(install_dir)

        rc = cli.run_list_models(args)

        out = capsys.readouterr().out
        assert rc == 0
        assert "2 model(s) cached locally" in out
        # The sized model carries no sizing complaint; the unsized one says so. Both are ungraded
        # here, so both carry the "?" marker and the benchmark prompt.
        assert "org/repo:Q4_K_M" in out
        assert "org/other:Q8_0  (not sized yet" in out
        assert out.count("not measured -- run `benchmark`") == 2
        assert "comparable to scores published there" in out  # the engine caveat is printed

    def test_no_presets_path_marks_nothing_sized(self, tmp_path, monkeypatch, capsys):
        install_dir = tmp_path / "install"
        write_runtime(install_dir, {"llama_cpp": {"server_binary": "/fake/llama-server"}, "service": {}})
        monkeypatch.setattr(model_ops, "list_cached_models", lambda binary: ["org/repo:Q4_K_M"])
        args = make_pull_args(install_dir)

        rc = cli.run_list_models(args)

        out = capsys.readouterr().out
        assert rc == 0
        assert "not sized yet" in out

    def test_no_cached_models_suggests_pull(self, tmp_path, monkeypatch, capsys):
        install_dir = tmp_path / "install"
        write_runtime(install_dir, {"llama_cpp": {"server_binary": "/fake/llama-server"}, "service": {}})
        monkeypatch.setattr(model_ops, "list_cached_models", lambda binary: [])
        args = make_pull_args(install_dir)

        rc = cli.run_list_models(args)

        out = capsys.readouterr().out
        assert rc == 0
        assert "aipotluck-local-client pull" in out

    def test_missing_llama_cpp_install_is_rejected(self, tmp_path, monkeypatch):
        install_dir = tmp_path / "install"
        write_runtime(install_dir, {"service": {}})  # no llama_cpp section
        calls = []
        monkeypatch.setattr(model_ops, "list_cached_models", lambda binary: calls.append(binary))
        args = make_pull_args(install_dir)

        rc = cli.run_list_models(args)

        assert rc == 1
        assert calls == []

    def test_list_error_is_surfaced_and_returns_1(self, tmp_path, monkeypatch, caplog):
        install_dir = tmp_path / "install"
        write_runtime(install_dir, {"llama_cpp": {"server_binary": "/fake/llama-server"}, "service": {}})

        def raise_error(binary):
            raise cli.ModelPullError("--cache-list not supported by this build")

        monkeypatch.setattr(model_ops, "list_cached_models", raise_error)
        args = make_pull_args(install_dir)

        with caplog.at_level(logging.ERROR, logger="aipotluck.cli"):
            rc = cli.run_list_models(args)

        assert rc == 1
        assert any("not supported" in rec.message for rec in caplog.records)


class TestDeleteCachedModel:
    """The service's own router is asked to unload the model before the files go, and the deletion
    itself goes to a scratch router (cache_router's own tests cover that half)."""

    def _serve_main_router(self):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        seen = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                seen.append((self.path, json.loads(body)))
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, seen

    def test_unloads_from_the_service_router_before_deleting(self, monkeypatch):
        server, seen = self._serve_main_router()
        order = []
        monkeypatch.setattr(
            model_ops.cache_router, "delete",
            lambda binary, mid: order.append(("delete", list(seen))),
        )
        try:
            model_ops.delete_cached_model(
                {"host": "127.0.0.1", "port": server.server_address[1],
                 "server_binary": "/fake/llama-server"},
                "org/repo:Q4_K_M",
            )
        finally:
            server.shutdown()
            server.server_close()
        assert order == [("delete", [("/models/unload", {"model": "org/repo:Q4_K_M"})])]

    def test_a_service_that_is_not_running_does_not_stop_the_delete(self, monkeypatch):
        deletes = []
        monkeypatch.setattr(model_ops.cache_router, "delete", lambda binary, mid: deletes.append(mid))
        model_ops.delete_cached_model(
            {"host": "127.0.0.1", "port": 1, "server_binary": "/fake/llama-server"}, "org/repo:Q4_K_M",
        )
        assert deletes == ["org/repo:Q4_K_M"]

    def test_a_refusal_is_reported_with_llama_cpps_reason(self, monkeypatch):
        def refuse(binary, mid):
            raise model_ops.cache_router.CacheRouterError("model name=x is not found")

        monkeypatch.setattr(model_ops.cache_router, "delete", refuse)
        with pytest.raises(model_ops.ModelOpError) as excinfo:
            model_ops.delete_cached_model(
                {"host": "127.0.0.1", "port": 1, "server_binary": "/fake/llama-server"}, "x",
            )
        assert excinfo.value.code == "delete_failed"
        assert "model name=x is not found" in str(excinfo.value)


def _screen(rejected, reason="test reason"):
    return cli.model_screen.ScreenResult(rejected, "too_slow" if rejected else None,
                                         reason, 1000.0, False, 0.5)


def _score(value=250.0, confidence="ok"):
    return cli.model_localscore.LocalScoreResult(value, 900.0, 30.0, 1200.0, "good", False,
                                                 confidence, 0.05)


class TestPullGate:
    """Sizing gates a pull, and a rejection deletes the model. Withholding a preset is not enough on
    its own: the router auto-discovers the HF cache, so a refused model would still be listed by
    /v1/models and selectable in the web app's picker. A pull never benchmarks -- that waits for
    an explicit `benchmark`."""

    def _setup(self, tmp_path, monkeypatch, screened, score):
        install_dir = tmp_path / "install"
        presets_path = tmp_path / "presets.ini"
        cli.model_presets.write_preset(presets_path, "org/repo:Q4_K_M", {"ctx-size": "32768"})
        write_runtime(install_dir, {
            "llama_cpp": {"server_binary": "/fake/llama-server", "presets_path": str(presets_path)},
            "service": {},
        })
        monkeypatch.setattr(model_ops, "pull_model", lambda *a, **kw: None)
        monkeypatch.setattr(model_ops.model_sizing, "ensure_preset", lambda *a, **kw: _FAKE_SIZING)
        monkeypatch.setattr(model_ops, "reload_router", lambda llama_cfg: True)
        monkeypatch.setattr(model_ops, "measure", lambda *a, **kw: (screened, score, None))
        deletes = []
        monkeypatch.setattr(model_ops, "delete_cached_model",
                            lambda cfg, mid: deletes.append(mid) or True)
        return install_dir, deletes

    def test_a_pull_never_benchmarks_and_says_how_to(self, tmp_path, monkeypatch, capsys):
        """Measuring made every pull minutes longer than its download, for a verdict nobody had
        asked for yet. Even a model the benchmark WOULD reject is kept: that verdict is the
        benchmark's to give, when it is asked for."""
        install_dir, deletes = self._setup(tmp_path, monkeypatch, _screen(True), None)
        benchmarks = []
        monkeypatch.setattr(model_ops, "measure",
                            lambda *a, **kw: benchmarks.append(1) or (_screen(True), None, None))

        rc = cli.run_pull_model(make_pull_args(install_dir))

        assert rc == 0 and benchmarks == [] and deletes == []
        assert "benchmark org/repo:Q4_K_M" in capsys.readouterr().out

    def test_the_old_skip_benchmark_flag_is_still_accepted(self):
        """Scripts written when it mattered must not start failing to parse."""
        args = cli.build_arg_parser().parse_args(["pull", "org/repo:Q4_K_M", "--skip-benchmark"])
        assert args.model == "org/repo:Q4_K_M"

    def test_a_rejected_models_preset_is_dropped_with_it(self, tmp_path, monkeypatch):
        unviable = SizingResult(
            ctx_size=512, parallel=1, cache_type_k=None, cache_type_v=None,
            tuning={"ctx_size": "test reason"}, viable=False,
            rejection_code=model_sizing.REJECT_NO_ROOM_FOR_CONTEXT, rejection="no room",
        )
        install_dir, _deletes = self._setup(tmp_path, monkeypatch, None, None)
        monkeypatch.setattr(model_ops.model_sizing, "ensure_preset", lambda *a, **kw: unviable)
        cli.run_pull_model(make_pull_args(install_dir))
        assert cli.model_presets.read_all(tmp_path / "presets.ini") == {}

    def test_a_model_with_no_room_for_context_is_rejected_without_benchmarking(
        self, tmp_path, monkeypatch, capsys
    ):
        """Sizing already knows the answer, so the benchmark must not run at all -- loading and
        measuring a model to confirm what the memory budget settled is pure cost to the user."""
        unviable = SizingResult(
            ctx_size=512, parallel=1, cache_type_k="q4_0", cache_type_v="q4_0",
            tuning={"ctx_size": "test reason"}, viable=False,
            rejection_code=model_sizing.REJECT_NO_ROOM_FOR_CONTEXT,
            rejection="this device has 4.00GB of memory and its 8.00GB of weights exceed the budget",
        )
        install_dir, deletes = self._setup(tmp_path, monkeypatch, _screen(False), _score(412.0))
        benchmarks = []
        monkeypatch.setattr(model_ops, "measure",
                            lambda *a, **kw: benchmarks.append(1) or (_screen(False), _score(412.0), None))
        monkeypatch.setattr(model_ops.model_sizing, "ensure_preset", lambda *a, **kw: unviable)

        rc = cli.run_pull_model(make_pull_args(install_dir))

        out = capsys.readouterr().out
        assert rc == 1 and deletes == ["org/repo:Q4_K_M"]
        assert benchmarks == [], "sizing already decided; the benchmark is wasted work"
        assert "exceed the budget" in out, "the reason must survive into the user-facing output"

    def test_force_keeps_a_model_sizing_would_reject(self, tmp_path, monkeypatch):
        unviable = SizingResult(
            ctx_size=512, parallel=1, cache_type_k=None, cache_type_v=None,
            tuning={"ctx_size": "test reason"}, viable=False,
            rejection_code=model_sizing.REJECT_NO_ROOM_FOR_CONTEXT, rejection="no room",
        )
        install_dir, deletes = self._setup(tmp_path, monkeypatch, _screen(False), _score(412.0))
        monkeypatch.setattr(model_ops.model_sizing, "ensure_preset", lambda *a, **kw: unviable)
        assert cli.run_pull_model(make_pull_args(install_dir, force=True)) == 0
        assert deletes == []


class TestRunBenchmark:
    """Where a model's speed verdict is shown now that pull no longer measures."""

    def _setup(self, tmp_path, monkeypatch, screened, score):
        install_dir = tmp_path / "install"
        presets_path = tmp_path / "presets.ini"
        cli.model_presets.write_preset(presets_path, "org/repo:Q4_K_M", {"ctx-size": "32768"})
        write_runtime(install_dir, {
            "llama_cpp": {"server_binary": "/fake/llama-server", "presets_path": str(presets_path)},
            "service": {},
        })
        monkeypatch.setattr(model_ops.model_sizing, "ensure_preset", lambda *a, **kw: _FAKE_SIZING)
        monkeypatch.setattr(model_ops, "measure", lambda *a, **kw: (screened, score, None))
        return install_dir

    def test_a_usable_model_reports_its_score(self, tmp_path, monkeypatch, capsys):
        install_dir = self._setup(tmp_path, monkeypatch, _screen(False), _score(412.0))
        rc = cli.run_benchmark(Namespace(install_dir=install_dir, system=False, verbose=False,
                                         model="org/repo:Q4_K_M"))
        assert rc == 0
        assert "LocalScore 412" in capsys.readouterr().out

    def test_a_low_confidence_score_says_so(self, tmp_path, monkeypatch, capsys):
        """A device that moved 30% under its own measurement cannot support a precise number, and
        the output has to admit that rather than printing it as though it could."""
        install_dir = self._setup(tmp_path, monkeypatch, _screen(False), _score(confidence="low"))
        cli.run_benchmark(Namespace(install_dir=install_dir, system=False, verbose=False,
                                    model="org/repo:Q4_K_M"))
        assert "low confidence" in capsys.readouterr().out
