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
from aipotluck.installer.model_sizing import SizingResult
from aipotluck.installer.platform_detect import HostProfile


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
            result = cli._reload_router_models({"host": "127.0.0.1", "port": port})
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

        assert cli._reload_router_models({"host": "127.0.0.1", "port": port}) is False

    def test_defaults_to_localhost_8080_when_unset(self, monkeypatch):
        # Must not raise with an empty llama_cfg -- falls back to llama-server's own conventional
        # default host/port rather than crashing on a missing key.
        seen_urls = []

        def fake_urlopen(url, timeout):
            seen_urls.append(url)
            return _FakeHttpResponse({})

        monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
        assert cli._reload_router_models({}) is True
        assert seen_urls == ["http://127.0.0.1:8080/models?reload=1"]


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
        monkeypatch.setattr(cli, "pull_model", lambda *a, **kw: pull_calls.append((a, kw)))
        ensure_preset_calls = []
        monkeypatch.setattr(cli, "ensure_preset", lambda *a, **kw: ensure_preset_calls.append((a, kw)) or _FAKE_SIZING)
        reload_calls = []
        monkeypatch.setattr(cli, "_reload_router_models", lambda llama_cfg: reload_calls.append(1) or True)
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
        monkeypatch.setattr(cli, "pull_model", lambda *a, **kw: pull_calls.append((a, kw)))
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

        monkeypatch.setattr(cli, "pull_model", raise_pull_error)
        ensure_preset_calls = []
        monkeypatch.setattr(cli, "ensure_preset", lambda *a, **kw: ensure_preset_calls.append(1))
        args = make_pull_args(install_dir, model="bad/repo")

        rc = cli.run_pull_model(args)

        assert rc == 1
        assert ensure_preset_calls == []

    def test_default_timeout_used_when_not_given(self, tmp_path, monkeypatch):
        install_dir = tmp_path / "install"
        write_runtime(install_dir, {"llama_cpp": {"server_binary": "/fake/llama-server"}, "service": {}})
        seen_kwargs = {}
        monkeypatch.setattr(cli, "pull_model", lambda *a, **kw: seen_kwargs.update(kw))
        monkeypatch.setattr(cli, "ensure_preset", lambda *a, **kw: _FAKE_SIZING)
        args = make_pull_args(install_dir, timeout=None)

        cli.run_pull_model(args)

        assert seen_kwargs["timeout"] == cli.DEFAULT_TIMEOUT_SECONDS

    def test_custom_timeout_passed_through(self, tmp_path, monkeypatch):
        install_dir = tmp_path / "install"
        write_runtime(install_dir, {"llama_cpp": {"server_binary": "/fake/llama-server"}, "service": {}})
        seen_kwargs = {}
        monkeypatch.setattr(cli, "pull_model", lambda *a, **kw: seen_kwargs.update(kw))
        monkeypatch.setattr(cli, "ensure_preset", lambda *a, **kw: _FAKE_SIZING)
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
        monkeypatch.setattr(cli, "pull_model", lambda *a, **kw: pull_kwargs.update(kw))
        ensure_preset_kwargs = {}
        monkeypatch.setattr(
            cli, "ensure_preset", lambda *a, **kw: ensure_preset_kwargs.update(kw) or _FAKE_SIZING
        )
        args = make_pull_args(install_dir)

        cli.run_pull_model(args)

        assert pull_kwargs["gpu_layers"] == "all"
        assert ensure_preset_kwargs["gpu_layers"] == "all"

    def test_no_presets_path_skips_sizing_but_still_succeeds(self, tmp_path, monkeypatch, caplog):
        install_dir = tmp_path / "install"
        write_runtime(install_dir, {"llama_cpp": {"server_binary": "/fake/llama-server"}, "service": {}})
        monkeypatch.setattr(cli, "pull_model", lambda *a, **kw: None)
        ensure_preset_calls = []
        monkeypatch.setattr(cli, "ensure_preset", lambda *a, **kw: ensure_preset_calls.append(1))
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
        monkeypatch.setattr(cli, "pull_model", lambda *a, **kw: None)

        def raise_sizing_error(*a, **kw):
            raise cli.ModelSizingError("probe timed out")

        monkeypatch.setattr(cli, "ensure_preset", raise_sizing_error)
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
        monkeypatch.setattr(cli, "pull_model", lambda *a, **kw: None)
        monkeypatch.setattr(cli, "ensure_preset", lambda *a, **kw: _FAKE_SIZING)
        monkeypatch.setattr(cli, "_reload_router_models", lambda llama_cfg: False)
        args = make_pull_args(install_dir)

        rc = cli.run_pull_model(args)

        assert rc == 0
        out = capsys.readouterr().out
        assert "next time it starts" in out


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
        monkeypatch.setattr(cli, "list_cached_models", lambda binary: ["org/repo:Q4_K_M", "org/other:Q8_0"])
        args = make_pull_args(install_dir)

        rc = cli.run_list_models(args)

        out = capsys.readouterr().out
        assert rc == 0
        assert "2 model(s) cached locally" in out
        # The sized model carries no sizing complaint; the unsized one says so. Both are ungraded
        # here, so both carry the "?" marker and the benchmark prompt.
        assert "org/repo:Q4_K_M" in out
        assert "org/repo:Q4_K_M  (not sized yet" not in out
        assert "org/other:Q8_0  (not sized yet" in out
        assert out.count("not benchmarked -- run `benchmark`") == 2

    def test_no_presets_path_marks_nothing_sized(self, tmp_path, monkeypatch, capsys):
        install_dir = tmp_path / "install"
        write_runtime(install_dir, {"llama_cpp": {"server_binary": "/fake/llama-server"}, "service": {}})
        monkeypatch.setattr(cli, "list_cached_models", lambda binary: ["org/repo:Q4_K_M"])
        args = make_pull_args(install_dir)

        rc = cli.run_list_models(args)

        out = capsys.readouterr().out
        assert rc == 0
        assert "not sized yet" in out

    def test_no_cached_models_suggests_pull(self, tmp_path, monkeypatch, capsys):
        install_dir = tmp_path / "install"
        write_runtime(install_dir, {"llama_cpp": {"server_binary": "/fake/llama-server"}, "service": {}})
        monkeypatch.setattr(cli, "list_cached_models", lambda binary: [])
        args = make_pull_args(install_dir)

        rc = cli.run_list_models(args)

        out = capsys.readouterr().out
        assert rc == 0
        assert "aipotluck-local-client pull" in out

    def test_missing_llama_cpp_install_is_rejected(self, tmp_path, monkeypatch):
        install_dir = tmp_path / "install"
        write_runtime(install_dir, {"service": {}})  # no llama_cpp section
        calls = []
        monkeypatch.setattr(cli, "list_cached_models", lambda binary: calls.append(binary))
        args = make_pull_args(install_dir)

        rc = cli.run_list_models(args)

        assert rc == 1
        assert calls == []

    def test_list_error_is_surfaced_and_returns_1(self, tmp_path, monkeypatch, caplog):
        install_dir = tmp_path / "install"
        write_runtime(install_dir, {"llama_cpp": {"server_binary": "/fake/llama-server"}, "service": {}})

        def raise_error(binary):
            raise cli.ModelPullError("--cache-list not supported by this build")

        monkeypatch.setattr(cli, "list_cached_models", raise_error)
        args = make_pull_args(install_dir)

        with caplog.at_level(logging.ERROR, logger="aipotluck.cli"):
            rc = cli.run_list_models(args)

        assert rc == 1
        assert any("not supported" in rec.message for rec in caplog.records)


def _graded(grade: str, n_fit: int = 40_000) -> "cli.model_perf.PerfResult":
    fit = cli.model_perf.PerfFit(
        decode_base_ms=20.0, decode_depth_ms=0.0, prefill_base_ms=2.0, prefill_depth_ms=0.0,
        load_ms=0.0, source=cli.model_perf.SOURCE_LLAMA_BENCH, confidence=cli.model_perf.CONFIDENCE_OK,
    )
    return cli.model_perf.PerfResult(
        n_fit=n_fit, grade=grade, decode_tokens_per_second=50.0, ctx_size=131072,
        fit=fit, reason="test reason",
    )


class TestPullSpeedGate:
    """The refusal half of the speed grade. Withholding a preset would NOT be enough on its own --
    the router auto-discovers the HF cache, so a refused model would still be listed by /v1/models
    and selectable in the web app's picker. Deleting the cached files is the enforcement, and these
    assert it actually happens."""

    def _setup(self, tmp_path, monkeypatch, result, presets_ini=True):
        install_dir = tmp_path / "install"
        presets_path = tmp_path / "presets.ini"
        if presets_ini:
            cli.model_presets.write_preset(presets_path, "org/repo:Q4_K_M", {"ctx-size": "131072"})
        write_runtime(
            install_dir,
            {
                "llama_cpp": {"server_binary": "/fake/llama-server", "presets_path": str(presets_path)},
                "service": {},
            },
        )
        monkeypatch.setattr(cli, "pull_model", lambda *a, **kw: None)
        monkeypatch.setattr(cli, "ensure_preset", lambda *a, **kw: _FAKE_SIZING)
        monkeypatch.setattr(cli, "_reload_router_models", lambda llama_cfg: True)
        monkeypatch.setattr(cli, "_benchmark_model", lambda *a, **kw: result)
        deletes = []
        monkeypatch.setattr(
            cli, "_delete_cached_model",
            lambda llama_cfg, model_id: deletes.append(model_id) or True,
        )
        return install_dir, deletes

    def test_a_refused_model_is_deleted_and_the_pull_fails(self, tmp_path, monkeypatch, capsys):
        install_dir, deletes = self._setup(tmp_path, monkeypatch, _graded(cli.model_perf.GRADE_REFUSE, 0))

        rc = cli.run_pull_model(make_pull_args(install_dir))

        assert rc == 1
        assert deletes == ["org/repo:Q4_K_M"]
        out = capsys.readouterr().out
        assert "too slow" in out
        assert "--force" in out

    def test_force_keeps_a_refused_model(self, tmp_path, monkeypatch, capsys):
        install_dir, deletes = self._setup(tmp_path, monkeypatch, _graded(cli.model_perf.GRADE_REFUSE, 0))

        rc = cli.run_pull_model(make_pull_args(install_dir, force=True))

        assert rc == 0
        assert deletes == []
        assert "REFUSE" in capsys.readouterr().out

    @pytest.mark.parametrize("grade", [
        cli.model_perf.GRADE_GREEN, cli.model_perf.GRADE_YELLOW, cli.model_perf.GRADE_RED,
    ])
    def test_a_usable_model_is_kept_and_its_grade_reported(self, tmp_path, monkeypatch, capsys, grade):
        """Red is a warning, not a refusal: it still works for short turns, and the user is told."""
        install_dir, deletes = self._setup(tmp_path, monkeypatch, _graded(grade, 5_000))

        rc = cli.run_pull_model(make_pull_args(install_dir))

        assert rc == 0
        assert deletes == []
        out = capsys.readouterr().out
        assert grade.upper() in out
        assert "5,000 tokens" in out

    def test_skip_benchmark_pulls_without_measuring(self, tmp_path, monkeypatch, capsys):
        install_dir, deletes = self._setup(tmp_path, monkeypatch, _graded(cli.model_perf.GRADE_REFUSE, 0))

        rc = cli.run_pull_model(make_pull_args(install_dir, skip_benchmark=True))

        assert rc == 0
        assert deletes == []
        assert "Measuring" not in capsys.readouterr().out

    def test_an_unmeasurable_model_is_kept_rather_than_refused_on_no_evidence(self, tmp_path, monkeypatch):
        """A missing grade must never become a refusal. Failing to measure says nothing about the
        model, and deleting someone's download on the strength of a failed probe would be worse
        than the problem this feature exists to solve."""
        install_dir, deletes = self._setup(tmp_path, monkeypatch, None)

        rc = cli.run_pull_model(make_pull_args(install_dir))

        assert rc == 0
        assert deletes == []

    def test_refusal_survives_an_unreachable_router_by_telling_the_user(self, tmp_path, monkeypatch, capsys):
        install_dir, _ = self._setup(tmp_path, monkeypatch, _graded(cli.model_perf.GRADE_REFUSE, 0))
        monkeypatch.setattr(cli, "_delete_cached_model", lambda llama_cfg, model_id: False)

        rc = cli.run_pull_model(make_pull_args(install_dir))

        assert rc == 1
        assert "still on disk" in capsys.readouterr().out


class TestBenchmarkModelSelection:
    """_benchmark_model's guards -- the cases where it must decline to produce a number."""

    def _cfg(self, tmp_path, *, ctx="131072", bench_exists=True):
        presets_path = tmp_path / "presets.ini"
        if ctx:
            cli.model_presets.write_preset(presets_path, "org/repo:Q4_K_M", {"ctx-size": ctx})
        binary = tmp_path / "llama-server"
        binary.write_text("#!/bin/sh\n", encoding="utf-8")
        if bench_exists:
            (tmp_path / "llama-bench").write_text("#!/bin/sh\n", encoding="utf-8")
        return {"server_binary": str(binary)}, presets_path

    def test_declines_when_the_model_has_no_sized_preset(self, tmp_path, monkeypatch, caplog):
        llama_cfg, presets_path = self._cfg(tmp_path, ctx=None)
        with caplog.at_level("WARNING"):
            result = cli._benchmark_model(llama_cfg, presets_path, "org/repo:Q4_K_M", budget_seconds=10)
        assert result is None
        assert "no sized preset" in caplog.text

    def test_declines_while_the_machine_is_busy(self, tmp_path, monkeypatch, caplog):
        llama_cfg, presets_path = self._cfg(tmp_path)
        monkeypatch.setattr(cli.model_perf, "machine_is_too_busy", lambda: True)
        with caplog.at_level("WARNING"):
            result = cli._benchmark_model(llama_cfg, presets_path, "org/repo:Q4_K_M", budget_seconds=10)
        assert result is None
        assert "already busy" in caplog.text

    def test_falls_back_to_the_live_server_when_llama_bench_is_absent(self, tmp_path, monkeypatch):
        """Source-built installs (Jetson-class arm64+CUDA, old glibc) have no llama-bench, and they
        are the hardware most likely to be too slow -- so they must still get measured."""
        llama_cfg, presets_path = self._cfg(tmp_path, bench_exists=False)
        monkeypatch.setattr(cli.model_perf, "machine_is_too_busy", lambda: False)
        live_calls = []
        monkeypatch.setattr(
            cli.model_perf_live, "probe_performance_live",
            lambda *a, **kw: live_calls.append((a, kw)) or _graded(cli.model_perf.GRADE_YELLOW),
        )
        monkeypatch.setattr(
            cli.model_perf, "probe_performance",
            lambda *a, **kw: pytest.fail("llama-bench path must not run when the binary is absent"),
        )

        result = cli._benchmark_model(llama_cfg, presets_path, "org/repo:Q4_K_M", budget_seconds=10)

        assert result is not None and len(live_calls) == 1
        assert live_calls[0][1]["ctx_size"] == 131072
        # The grade is persisted so `list` can show it without re-measuring.
        assert cli.model_perf_store.read_record(presets_path, "org/repo:Q4_K_M")["grade"] == "yellow"

    def test_prefers_llama_bench_when_it_is_present(self, tmp_path, monkeypatch):
        llama_cfg, presets_path = self._cfg(tmp_path, bench_exists=True)
        monkeypatch.setattr(cli.model_perf, "machine_is_too_busy", lambda: False)
        bench_calls = []
        monkeypatch.setattr(
            cli.model_perf, "probe_performance",
            lambda *a, **kw: bench_calls.append(kw) or _graded(cli.model_perf.GRADE_GREEN),
        )
        monkeypatch.setattr(
            cli.model_perf_live, "probe_performance_live",
            lambda *a, **kw: pytest.fail("live fallback must not run when llama-bench exists"),
        )

        cli._benchmark_model(llama_cfg, presets_path, "org/repo:Q4_K_M", budget_seconds=10)

        assert bench_calls[0]["cache_type_k"] is None or "cache_type_k" in bench_calls[0]
        assert bench_calls[0]["ctx_size"] == 131072


class TestDeleteCachedModel:
    """Real HTTP: proves the exact request llama.cpp's router expects, since this is the only thing
    that actually removes a refused model. The route is `DELETE /models?model=<id>`
    (vendor/llama.cpp/tools/server/server.cpp's `ctx_http.del("/models", ...)` ->
    server-models.cpp's `del_router_models`, which reads the `model` QUERY parameter -- not a JSON
    body -- and calls common_download_remove to clear the snapshot, its symlinks and the orphaned
    blobs). Getting the method or the parameter shape wrong would silently leave the model in place
    while the CLI reported it removed."""

    def _serve(self, status=200):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        seen = []

        class Handler(BaseHTTPRequestHandler):
            def do_DELETE(self):
                seen.append((self.command, self.path))
                self.send_response(status)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, seen

    def test_sends_a_delete_with_the_model_as_a_url_encoded_query_parameter(self):
        server, seen = self._serve()
        try:
            port = server.server_address[1]
            ok = cli._delete_cached_model({"host": "127.0.0.1", "port": port}, "org/repo:Q4_K_M")
        finally:
            server.shutdown()
            server.server_close()

        assert ok is True
        assert len(seen) == 1
        method, path = seen[0]
        assert method == "DELETE"
        # The id contains both "/" and ":", so it has to survive encoding intact.
        assert path == "/models?model=org%2Frepo%3AQ4_K_M"

    def test_an_unreachable_router_reports_failure_rather_than_claiming_success(self):
        assert cli._delete_cached_model({"host": "127.0.0.1", "port": 1}, "org/repo:Q4_K_M") is False
