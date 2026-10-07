"""aipotluck.service.runner -- the portable service body.

The two things this file exists to pin: (1) both supervisors are gated on runtime.json's
"logged_in" flag, and (2) /status never echoes the tunnel secret. Real LlamaSupervisor/
NewtSupervisor are swapped for lightweight fakes throughout -- nothing here spawns a real
subprocess.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

import pytest

from aipotluck.service import runner


class _FakeSupervisor:
    """Records start()/stop() calls; stands in for LlamaSupervisor/NewtSupervisor so tests never
    spawn a real llama-server or newt process."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self, timeout: float = 20.0):
        self.stopped = True

    def info(self):
        from types import SimpleNamespace
        return SimpleNamespace(running=self.started)


class TestLoadRuntimeConfig:
    def test_missing_file_returns_empty_dict(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="aipotluck.service"):
            result = runner.load_runtime_config(tmp_path)
        assert result == {}
        assert any("run the installer first" in rec.message for rec in caplog.records)

    def test_valid_json_is_parsed(self, tmp_path):
        (tmp_path / "runtime.json").write_text(json.dumps({"logged_in": True}), encoding="utf-8")
        assert runner.load_runtime_config(tmp_path) == {"logged_in": True}

    def test_invalid_json_returns_empty_dict_not_raises(self, tmp_path, caplog):
        (tmp_path / "runtime.json").write_text("{not valid json", encoding="utf-8")
        with caplog.at_level(logging.ERROR, logger="aipotluck.service"):
            result = runner.load_runtime_config(tmp_path)
        assert result == {}
        assert any("Failed to read runtime.json" in rec.message for rec in caplog.records)


class TestBuildLlamaServerArgs:
    """Router mode (CUR-1965's follow-up): no -hf/--model/--ctx-size/--parallel/--cache-type-k/-v
    of our own -- those are per-model now, set in the --models-preset INI file instead (see
    test_model_presets.py/test_model_sizing.py). This just covers the router-level args."""

    def test_host_and_port_passed_through(self):
        args = runner.build_llama_server_args({"host": "0.0.0.0", "port": 9090})
        assert "--host" in args and "0.0.0.0" in args
        assert "--port" in args and "9090" in args

    def test_presets_path_passed_through_when_set(self):
        args = runner.build_llama_server_args({"host": "127.0.0.1", "port": 8080, "presets_path": "/cfg/presets.ini"})
        assert "--models-preset" in args and "/cfg/presets.ini" in args

    def test_warns_when_no_presets_path_given(self, caplog):
        with caplog.at_level(logging.WARNING, logger="aipotluck.service"):
            runner.build_llama_server_args({"host": "127.0.0.1", "port": 8080})
        assert any("No presets_path configured" in rec.message for rec in caplog.records)

    def test_models_dir_passed_through_when_set(self):
        args = runner.build_llama_server_args({"host": "127.0.0.1", "port": 8080, "models_dir": "/models"})
        assert "--models-dir" in args and "/models" in args

    def test_models_dir_omitted_when_unset(self):
        args = runner.build_llama_server_args({"host": "127.0.0.1", "port": 8080})
        assert "--models-dir" not in args

    def test_models_max_defaults_to_one(self):
        args = runner.build_llama_server_args({"host": "127.0.0.1", "port": 8080})
        idx = args.index("--models-max")
        assert args[idx + 1] == "1"

    def test_models_max_overridable(self):
        args = runner.build_llama_server_args({"host": "127.0.0.1", "port": 8080, "models_max": 3})
        idx = args.index("--models-max")
        assert args[idx + 1] == "3"

    def test_gpu_layers_passed_through_as_the_one_global_model_arg(self):
        # Confirmed against server-models.cpp: the router overlays its own CLI args onto every
        # model instance, which is exactly why gpu_layers (a hardware capability, not a per-model
        # choice) stays here rather than moving into the per-model preset.
        args = runner.build_llama_server_args({"host": "127.0.0.1", "port": 8080, "gpu_layers": "auto"})
        assert "--gpu-layers" in args and "auto" in args

    def test_gpu_layers_omitted_when_unset(self):
        args = runner.build_llama_server_args({"host": "127.0.0.1", "port": 8080})
        assert "--gpu-layers" not in args

    def test_no_model_or_ctx_size_or_parallel_or_cache_type_flags_are_ever_emitted(self):
        # These would be a regression back to single-model mode -- router mode must never see them.
        args = runner.build_llama_server_args(
            {
                "host": "127.0.0.1", "port": 8080, "model_hf": "org/repo", "model_path": "/x.gguf",
                "ctx_size": 4096, "parallel": 1, "cache_type_k": "q8_0", "cache_type_v": "q8_0",
            }
        )
        for flag in ("-hf", "--model", "--ctx-size", "--parallel", "--cache-type-k", "--cache-type-v"):
            assert flag not in args, f"{flag} must not be emitted in router mode"


class TestBackfillMissingPresets:
    """runner.backfill_missing_presets orchestrates three independently-tested collaborators
    (model_pull.list_cached_models, model_presets.known_model_ids, model_sizing.ensure_preset --
    each has real-subprocess/real-file coverage of its own in test_model_pull.py/
    test_model_presets.py/test_model_sizing.py) -- what belongs here is the orchestration logic
    itself: which models get sized, and that this never blocks/crashes service startup."""

    def test_no_op_when_server_binary_missing_from_config(self, monkeypatch):
        calls = []
        monkeypatch.setattr(runner.model_pull, "list_cached_models", lambda *a, **kw: calls.append(1))
        runner.backfill_missing_presets({"presets_path": "/cfg/presets.ini"})
        assert calls == []

    def test_no_op_when_presets_path_missing_from_config(self, monkeypatch):
        calls = []
        monkeypatch.setattr(runner.model_pull, "list_cached_models", lambda *a, **kw: calls.append(1))
        runner.backfill_missing_presets({"server_binary": "/fake/llama-server"})
        assert calls == []

    def test_no_op_when_nothing_is_missing(self, monkeypatch):
        monkeypatch.setattr(runner.model_pull, "list_cached_models", lambda *a, **kw: ["org/a:Q4", "org/b:Q8"])
        monkeypatch.setattr(runner.model_presets, "known_model_ids", lambda *a, **kw: {"org/a:Q4", "org/b:Q8"})
        ensure_calls = []
        monkeypatch.setattr(runner.model_sizing, "ensure_preset", lambda *a, **kw: ensure_calls.append(kw))
        runner.backfill_missing_presets({"server_binary": "/fake/llama-server", "presets_path": "/cfg/presets.ini"})
        assert ensure_calls == []

    def test_sizes_every_cached_model_missing_a_preset(self, monkeypatch):
        monkeypatch.setattr(runner.model_pull, "list_cached_models", lambda *a, **kw: ["org/a:Q4", "org/b:Q8"])
        monkeypatch.setattr(runner.model_presets, "known_model_ids", lambda *a, **kw: set())
        ensure_calls = []
        monkeypatch.setattr(runner.model_sizing, "ensure_preset", lambda *a, **kw: ensure_calls.append(kw))
        runner.backfill_missing_presets(
            {"server_binary": "/fake/llama-server", "presets_path": "/cfg/presets.ini", "gpu_layers": "auto"}
        )
        assert {c["model_hf"] for c in ensure_calls} == {"org/a:Q4", "org/b:Q8"}
        assert all(c["gpu_layers"] == "auto" for c in ensure_calls)

    def test_a_sizing_failure_for_one_model_does_not_stop_the_rest(self, monkeypatch):
        monkeypatch.setattr(runner.model_pull, "list_cached_models", lambda *a, **kw: ["org/bad:Q4", "org/good:Q8"])
        monkeypatch.setattr(runner.model_presets, "known_model_ids", lambda *a, **kw: set())
        sized = []

        def fake_ensure_preset(*a, **kw):
            if kw["model_hf"] == "org/bad:Q4":
                raise runner.model_sizing.ModelSizingError("probe failed")
            sized.append(kw["model_hf"])

        monkeypatch.setattr(runner.model_sizing, "ensure_preset", fake_ensure_preset)
        runner.backfill_missing_presets({"server_binary": "/fake/llama-server", "presets_path": "/cfg/presets.ini"})
        assert sized == ["org/good:Q8"]

    def test_list_cached_models_failure_is_a_no_op_not_a_crash(self, monkeypatch):
        def raise_pull_error(*a, **kw):
            raise runner.model_pull.ModelPullError("no server binary")

        monkeypatch.setattr(runner.model_pull, "list_cached_models", raise_pull_error)
        # Must not raise.
        runner.backfill_missing_presets({"server_binary": "/fake/llama-server", "presets_path": "/cfg/presets.ini"})


class TestEnsurePresetsFile:
    def test_creates_an_empty_file_where_none_exists(self, tmp_path):
        """llama-server will not start in router mode without it -- confirmed against b10989."""
        presets = tmp_path / "config" / "llama_presets.ini"
        runner.ensure_presets_file({"presets_path": str(presets)})
        assert presets.read_text(encoding="utf-8") == ""

    def test_never_touches_an_existing_file(self, tmp_path):
        presets = tmp_path / "llama_presets.ini"
        presets.write_text("[org/a:Q4]\nctx-size = 4096\n", encoding="utf-8")
        runner.ensure_presets_file({"presets_path": str(presets)})
        assert presets.read_text(encoding="utf-8") == "[org/a:Q4]\nctx-size = 4096\n"

    def test_no_presets_path_is_a_no_op(self):
        runner.ensure_presets_file({})


class TestBackfillJob:
    def _stub(self, monkeypatch, missing, ensure_preset):
        monkeypatch.setattr(runner, "models_missing_presets", lambda cfg: list(missing))
        monkeypatch.setattr(runner.model_sizing, "ensure_preset", ensure_preset)
        order = []
        monkeypatch.setattr(runner.model_perf_live, "free_router_memory", lambda url: order.append("free"))
        monkeypatch.setattr(runner.model_ops, "reload_router", lambda cfg: order.append("reload") or True)
        return order

    def test_frees_the_router_before_each_probe_and_reloads_once_at_the_end(self, monkeypatch):
        """The router is already serving by the time this runs, so a probe would otherwise share
        memory with whatever it holds -- and new presets do nothing until the router reloads."""
        order = []
        self_order = self._stub(monkeypatch, ["org/a:Q4", "org/b:Q8"],
                                lambda *a, **kw: order.append(("size", kw["model_hf"])))
        summary = runner.backfill_missing_presets({"server_binary": "/x", "presets_path": "/p"})
        assert summary == {"sized": ["org/a:Q4", "org/b:Q8"], "failed": {}, "router_reloaded": True}
        assert self_order == ["free", "free", "reload"]

    def test_a_failure_is_reported_in_the_summary_not_raised(self, monkeypatch):
        def ensure_preset(*a, **kw):
            raise runner.model_sizing.ModelSizingError("probe failed")

        order = self._stub(monkeypatch, ["org/a:Q4"], ensure_preset)
        summary = runner.backfill_missing_presets({"server_binary": "/x", "presets_path": "/p"})
        assert summary == {"sized": [], "failed": {"org/a:Q4": "probe failed"}, "router_reloaded": False}
        assert "reload" not in order


class TestRuntimeParamsWiring:
    """runtime_params' own logic (field extraction, tuning passthrough) is tested where it's
    actually defined -- test_diagnostics.py. What belongs here is proving GET /status really
    calls the real, shared implementation (runner.py imports it from diagnostics.py rather than
    keeping a second copy -- see CLAUDE.md's "Runtime parameters" convention for why that matters)."""

    def test_result_flows_into_the_real_status_endpoint(self, tmp_path):
        # Not mocked -- a real HTTP GET against a real running server, and a real presets INI +
        # tuning JSON on disk, proving runtime_params is actually wired into /status end to end
        # (router-level fields AND the per-model view it reads off disk), not just correct in
        # isolation (test_diagnostics.py already covers runtime_params' own logic in isolation).
        import json
        import urllib.request

        from aipotluck.installer import model_presets

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        presets_path = config_dir / "presets.ini"
        model_presets.write_preset(presets_path, "org/repo:Q4_K_M", {"ctx-size": "16384"})
        model_presets.write_tuning(presets_path, "org/repo:Q4_K_M", {"ctx_size": "auto: test"})
        (config_dir / "runtime.json").write_text(
            json.dumps({"logged_in": False, "llama_cpp": {"presets_path": str(presets_path)}}),
            encoding="utf-8",
        )
        svc = runner.AipotluckServiceRunner(host="127.0.0.1", port=0, config_dir=config_dir, log_dir=None)
        try:
            svc.start()
            port = svc._server.server_address[1]
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=5) as resp:
                body = json.loads(resp.read())
        finally:
            svc.stop(timeout=2)

        model_params = body["runtime_params"]["models"]["org/repo:Q4_K_M"]
        assert model_params["ctx_size"] == "16384"
        assert model_params["tuning"] == {"ctx_size": "auto: test"}


class TestHeldBackModelsInStatus:
    """A real HTTP GET against a real running server. The web app is where a model gets picked, so
    it is the only surface that can explain, at the moment a user chooses one, why it keeps
    failing -- which means the quarantine has to leave the CLI and reach /status."""

    def _status(self, config_dir):
        import json
        import urllib.request

        svc = runner.AipotluckServiceRunner(
            host="127.0.0.1", port=0, config_dir=config_dir, log_dir=None
        )
        try:
            svc.start()
            port = svc._server.server_address[1]
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=5) as resp:
                return json.loads(resp.read())
        finally:
            svc.stop(timeout=2)

    def _config(self, tmp_path):
        import json

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "runtime.json").write_text(
            json.dumps({"logged_in": False, "llama_cpp": {}}), encoding="utf-8"
        )
        return config_dir

    def test_a_held_back_model_reaches_the_status_endpoint_with_its_reason(self, tmp_path):
        from aipotluck.service import model_health

        config_dir = self._config(tmp_path)
        model_health.quarantine(config_dir, "org/repo:Q4_K_M", "ran out of memory 3 times")

        body = self._status(config_dir)

        assert "org/repo:Q4_K_M" in body["held_back_models"]
        assert body["held_back_models"]["org/repo:Q4_K_M"]["quarantine_reason"] == (
            "ran out of memory 3 times"
        )

    def test_the_field_is_present_and_empty_when_nothing_is_held_back(self, tmp_path):
        """Present-and-empty rather than absent, so a consumer can tell "nothing held back" from
        "this client is too old to report it"."""
        assert self._status(self._config(tmp_path))["held_back_models"] == {}


class TestBuildSupervisor:
    def test_returns_none_without_llama_cpp_section(self, caplog):
        with caplog.at_level(logging.ERROR, logger="aipotluck.service"):
            result = runner.build_supervisor({}, None)
        assert result is None
        assert any("no llama_cpp section" in rec.message for rec in caplog.records)

    def test_returns_none_when_binary_missing(self, tmp_path, caplog):
        config = {"llama_cpp": {"server_binary": str(tmp_path / "does-not-exist")}}
        with caplog.at_level(logging.ERROR, logger="aipotluck.service"):
            result = runner.build_supervisor(config, None)
        assert result is None
        assert any("binary not found" in rec.message for rec in caplog.records)

    def test_builds_supervisor_when_binary_exists(self, tmp_path, monkeypatch):
        binary = tmp_path / "llama-server"
        binary.write_text("#!/bin/sh\n")
        monkeypatch.setattr(runner, "LlamaSupervisor", _FakeSupervisor)

        config = {"llama_cpp": {"server_binary": str(binary), "host": "127.0.0.1", "port": 8080, "model_hf": "x"}}
        result = runner.build_supervisor(config, None)

        assert isinstance(result, _FakeSupervisor)


class TestBuildNewtSupervisor:
    def test_returns_none_without_tunnel_section(self):
        assert runner.build_newt_supervisor({}, None) is None

    def test_returns_none_when_binary_missing(self, tmp_path, caplog):
        config = {"tunnel": {"binary": str(tmp_path / "missing-newt")}}
        with caplog.at_level(logging.ERROR, logger="aipotluck.service"):
            result = runner.build_newt_supervisor(config, None)
        assert result is None
        assert any("newt binary not found" in rec.message for rec in caplog.records)

    def test_builds_supervisor_when_binary_exists(self, tmp_path, monkeypatch):
        binary = tmp_path / "newt"
        binary.write_text("#!/bin/sh\n")
        monkeypatch.setattr(runner, "NewtSupervisor", _FakeSupervisor)

        config = {"tunnel": {"binary": str(binary), "id": "i", "secret": "s", "endpoint": "e"}}
        result = runner.build_newt_supervisor(config, None)

        assert isinstance(result, _FakeSupervisor)


class TestRedactedRuntimeConfig:
    def test_redacts_tunnel_secret(self):
        config = {"tunnel": {"id": "i", "secret": "super-secret", "endpoint": "e"}, "logged_in": True}
        redacted = runner._StatusHandler._redacted_runtime_config(config)
        assert redacted["tunnel"]["secret"] == "<redacted>"
        assert redacted["tunnel"]["id"] == "i"
        # original must be untouched -- a caller holding onto `config` should never see it mutated
        assert config["tunnel"]["secret"] == "super-secret"

    def test_passthrough_when_no_tunnel_section(self):
        config = {"logged_in": False}
        assert runner._StatusHandler._redacted_runtime_config(config) == config


class TestCapabilitiesEndpoint:
    """A real HTTP GET against a real running server -- diagnostics.gather_fingerprint itself is
    faked (it's covered for real in test_diagnostics.py); what this pins is that the route is
    actually wired up and returns exactly what gather_fingerprint produced, as real JSON over the
    wire."""

    def test_returns_the_fingerprint_as_json(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "runtime.json").write_text(json.dumps({"logged_in": False}), encoding="utf-8")
        fake_fingerprint = {"service": "aipotluck", "fingerprint_version": 1, "os": {"name": "linux"}}
        monkeypatch.setattr(runner.diagnostics, "gather_fingerprint", lambda **kwargs: fake_fingerprint)

        svc = runner.AipotluckServiceRunner(host="127.0.0.1", port=0, config_dir=config_dir, log_dir=None)
        try:
            svc.start()
            port = svc._server.server_address[1]
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/capabilities", timeout=5) as resp:
                assert resp.status == 200
                assert resp.headers["Content-Type"] == "application/json"
                body = json.loads(resp.read())
            assert body == fake_fingerprint
        finally:
            svc.stop(timeout=2)

    def test_passes_config_dir_and_runtime_config_through(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "runtime.json").write_text(json.dumps({"logged_in": True}), encoding="utf-8")
        captured = {}

        def _fake_gather(*, config_dir, runtime_config):
            captured["config_dir"] = config_dir
            captured["runtime_config"] = runtime_config
            return {}

        monkeypatch.setattr(runner.diagnostics, "gather_fingerprint", _fake_gather)

        svc = runner.AipotluckServiceRunner(host="127.0.0.1", port=0, config_dir=config_dir, log_dir=None)
        try:
            svc.start()
            port = svc._server.server_address[1]
            urllib.request.urlopen(f"http://127.0.0.1:{port}/capabilities", timeout=5).read()
        finally:
            svc.stop(timeout=2)

        assert captured["config_dir"] == config_dir
        assert captured["runtime_config"]["logged_in"] is True


class TestAipotluckServiceRunnerStartStop:
    """Exercises the real start()/stop() methods end to end, including the real HTTP server --
    the only things swapped out are LlamaSupervisor/NewtSupervisor themselves."""

    def test_logged_out_start_binds_http_server_but_no_supervisors(self, tmp_path, monkeypatch, caplog):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "runtime.json").write_text(json.dumps({"logged_in": False}), encoding="utf-8")

        svc = runner.AipotluckServiceRunner(host="127.0.0.1", port=0, config_dir=config_dir, log_dir=None)
        try:
            with caplog.at_level(logging.INFO, logger="aipotluck.service"):
                svc.start()
            assert svc.supervisor is None
            assert svc.newt_supervisor is None
            assert svc._server is not None
            assert any("logged out" in rec.message for rec in caplog.records)
        finally:
            svc.stop(timeout=2)

    def test_logged_in_start_builds_and_starts_both_supervisors(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        binary = tmp_path / "llama-server"
        binary.write_text("#!/bin/sh\n")
        newt_binary = tmp_path / "newt"
        newt_binary.write_text("#!/bin/sh\n")
        (config_dir / "runtime.json").write_text(
            json.dumps(
                {
                    "logged_in": True,
                    "llama_cpp": {"server_binary": str(binary), "host": "127.0.0.1", "port": 8080, "model_hf": "x"},
                    "tunnel": {"binary": str(newt_binary), "id": "i", "secret": "s", "endpoint": "e"},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(runner, "LlamaSupervisor", _FakeSupervisor)
        monkeypatch.setattr(runner, "NewtSupervisor", _FakeSupervisor)

        svc = runner.AipotluckServiceRunner(host="127.0.0.1", port=0, config_dir=config_dir, log_dir=None)
        try:
            svc.start()
            assert isinstance(svc.supervisor, _FakeSupervisor)
            assert svc.supervisor.started
            assert isinstance(svc.newt_supervisor, _FakeSupervisor)
            assert svc.newt_supervisor.started
        finally:
            svc.stop(timeout=2)

    def _logged_in_config(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        binary = tmp_path / "llama-server"
        binary.write_text("#!/bin/sh\n")
        (config_dir / "runtime.json").write_text(
            json.dumps({
                "logged_in": True,
                "llama_cpp": {"server_binary": str(binary), "presets_path": str(tmp_path / "p.ini")},
            }),
            encoding="utf-8",
        )
        return config_dir

    def test_status_answers_while_a_model_is_still_being_sized(self, tmp_path, monkeypatch):
        """Sizing loads each model, which takes seconds to minutes. `status` and the model list
        must work from the moment the service starts, and the router must already be up."""
        import threading
        import time

        config_dir = self._logged_in_config(tmp_path)
        monkeypatch.setattr(runner, "LlamaSupervisor", _FakeSupervisor)
        monkeypatch.setattr(runner, "models_missing_presets", lambda cfg: ["org/unsized:Q4"])
        monkeypatch.setattr(runner.model_perf_live, "free_router_memory", lambda url: [])
        monkeypatch.setattr(runner.model_ops, "reload_router", lambda cfg: True)
        probing, release = threading.Event(), threading.Event()

        def slow_probe(*a, **kw):
            probing.set()
            release.wait(timeout=30)

        monkeypatch.setattr(runner.model_sizing, "ensure_preset", slow_probe)

        svc = runner.AipotluckServiceRunner(host="127.0.0.1", port=0, config_dir=config_dir, log_dir=None)
        try:
            started = time.monotonic()
            svc.start()
            assert time.monotonic() - started < 5
            assert probing.wait(timeout=5), "the backfill never started"
            assert svc.supervisor.started

            port = svc._server.server_address[1]
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=5) as resp:
                assert resp.status == 200
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/jobs", timeout=5) as resp:
                listed = json.loads(resp.read())["jobs"]
            assert [(j["kind"], j["state"]) for j in listed] == [("backfill", "running")]
        finally:
            release.set()
            svc.stop(timeout=2)

    def test_no_backfill_job_when_every_model_has_a_preset(self, tmp_path, monkeypatch):
        config_dir = self._logged_in_config(tmp_path)
        monkeypatch.setattr(runner, "LlamaSupervisor", _FakeSupervisor)
        monkeypatch.setattr(runner, "models_missing_presets", lambda cfg: [])
        svc = runner.AipotluckServiceRunner(host="127.0.0.1", port=0, config_dir=config_dir, log_dir=None)
        try:
            svc.start()
            assert svc.jobs.list() == []
        finally:
            svc.stop(timeout=2)

    def test_stop_is_idempotent(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "runtime.json").write_text(json.dumps({"logged_in": False}), encoding="utf-8")

        svc = runner.AipotluckServiceRunner(host="127.0.0.1", port=0, config_dir=config_dir, log_dir=None)
        svc.start()
        svc.stop(timeout=2)
        svc.stop(timeout=2)  # must not raise the second time
