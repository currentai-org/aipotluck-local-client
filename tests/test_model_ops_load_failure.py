"""What happens when a model cannot be loaded at all.

Sizing has to load the model to read its hparams, so a model too big for the device crashes the
probe -- which means the memory-budget check in compute_sizing can never run for exactly the models
that need it most. Before this, such a model stayed installed and unsized: the router discovers the
HF cache for itself, picks llama-server's defaults, and the failure resurfaces as a dead turn much
later with nothing connecting it back.

Found live on a Jetson with a 24B model whose 13.35GB of weights and 0.82GB vision projector did
not fit in 15.6GB of unified memory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aipotluck.installer import llama_errors, model_screen, model_sizing
from aipotluck.service import model_health, model_ops


def _load_failure(kind=llama_errors.FAILURE_OOM, exit_code=-6):
    return model_sizing.ModelSizingError(
        "llama-server exited (code=-6) while probing model metadata:\ncudaMalloc failed: out of memory",
        model_failed_to_load=True, failure_kind=kind, exit_code=exit_code,
    )


def _probe_failure():
    """A failure of the PROBE rather than of the model -- no binary, say. Must not reject."""
    return model_sizing.ModelSizingError("llama-server binary not found at /fake/llama-server")


@pytest.fixture
def env(tmp_path, monkeypatch):
    presets = tmp_path / "presets.ini"
    presets.write_text("", encoding="utf-8")
    llama_cfg = {
        "server_binary": "/fake/llama-server", "presets_path": str(presets),
        "host": "127.0.0.1", "port": 1,
    }
    freed = []
    monkeypatch.setattr(model_ops, "_free_router_before_loading", lambda cfg: freed.append(1))
    monkeypatch.setattr(model_ops, "pull_model", lambda *a, **kw: None)
    monkeypatch.setattr(model_ops, "list_cached_models", lambda binary: ["org/m:Q4_K_M"])
    monkeypatch.setattr(model_ops, "check_size_before_download", lambda m: None)
    monkeypatch.setattr(model_ops, "reload_router", lambda cfg: True)
    deletes = []
    monkeypatch.setattr(
        model_ops, "delete_cached_model", lambda cfg, mid: deletes.append(mid) or True
    )
    return llama_cfg, tmp_path, deletes, freed


class TestPullRejectsAModelThatCannotLoad:
    def test_an_oom_during_sizing_rejects_and_deletes(self, env, monkeypatch):
        llama_cfg, config_dir, deletes, _freed = env
        monkeypatch.setattr(
            model_ops.model_sizing, "ensure_preset",
            lambda *a, **kw: (_ for _ in ()).throw(_load_failure()),
        )

        result = model_ops.pull(llama_cfg, config_dir, "org/m:Q4_K_M")

        assert result["rejected"] is True
        assert result["rejection_code"] == model_screen.REJECT_WONT_LOAD
        assert "ran out of memory while loading" in result["rejection"]
        assert deletes == ["org/m:Q4_K_M"]

    def test_the_reason_names_the_vision_projector_too(self, env, monkeypatch):
        """The Jetson case failed allocating the mmproj, not the weights, and a user comparing the
        model's file size against their RAM would not understand why it did not fit."""
        llama_cfg, config_dir, _deletes, _freed = env
        monkeypatch.setattr(
            model_ops.model_sizing, "ensure_preset",
            lambda *a, **kw: (_ for _ in ()).throw(_load_failure()),
        )
        result = model_ops.pull(llama_cfg, config_dir, "org/m:Q4_K_M")
        assert "vision projector" in result["rejection"]

    def test_a_non_oom_load_failure_also_rejects(self, env, monkeypatch):
        """A model that aborts on load cannot serve either, whatever the reason."""
        llama_cfg, config_dir, deletes, _freed = env
        monkeypatch.setattr(
            model_ops.model_sizing, "ensure_preset",
            lambda *a, **kw: (_ for _ in ()).throw(_load_failure(kind=llama_errors.FAILURE_OTHER, exit_code=1)),
        )
        result = model_ops.pull(llama_cfg, config_dir, "org/m:Q4_K_M")
        assert result["rejected"] is True and deletes == ["org/m:Q4_K_M"]
        assert "exit code 1" in result["rejection"]

    def test_a_probe_failure_does_not_reject_the_model(self, env, monkeypatch):
        """The distinction that keeps this safe: a missing binary or an unparseable probe is the
        probe's problem. Deleting a model over it would destroy data for a local misconfiguration."""
        llama_cfg, config_dir, deletes, _freed = env
        monkeypatch.setattr(
            model_ops.model_sizing, "ensure_preset",
            lambda *a, **kw: (_ for _ in ()).throw(_probe_failure()),
        )
        monkeypatch.setattr(model_ops, "measure", lambda *a, **kw: (None, None, "no preset"))

        result = model_ops.pull(llama_cfg, config_dir, "org/m:Q4_K_M")

        assert not result.get("rejected")
        assert deletes == []
        assert "Automatic runtime sizing failed" in result["warning"]

    def test_keep_rejected_overrides_the_rejection(self, env, monkeypatch):
        llama_cfg, config_dir, deletes, _freed = env
        monkeypatch.setattr(
            model_ops.model_sizing, "ensure_preset",
            lambda *a, **kw: (_ for _ in ()).throw(_load_failure()),
        )
        monkeypatch.setattr(model_ops, "measure", lambda *a, **kw: (None, None, "no preset"))

        result = model_ops.pull(llama_cfg, config_dir, "org/m:Q4_K_M", keep_rejected=True)

        assert not result.get("rejected") and deletes == []

    def test_the_router_is_emptied_before_the_probe_loads_anything(self, env, monkeypatch):
        """Under --models-max 1 the router holds a whole model resident, so a probe starting then
        competes with it. This mattered less when a failure was a warning; now that it deletes, a
        model that works could be destroyed because the router happened to be busy."""
        llama_cfg, config_dir, _deletes, freed = env
        monkeypatch.setattr(model_ops.model_sizing, "ensure_preset", lambda *a, **kw: None)
        monkeypatch.setattr(model_ops, "measure", lambda *a, **kw: (None, None, "skip"))

        model_ops.pull(llama_cfg, config_dir, "org/m:Q4_K_M")

        assert freed == [1]


class TestBenchmarkHoldsBackAModelThatCannotLoad:
    def test_it_is_held_back_rather_than_deleted(self, env, monkeypatch):
        """The model is already installed and the user is standing right here having asked for a
        measurement, so the gigabytes are theirs to decide about -- the same split the service's
        own quarantine makes."""
        llama_cfg, config_dir, deletes, _freed = env
        monkeypatch.setattr(
            model_ops.model_sizing, "ensure_preset",
            lambda *a, **kw: (_ for _ in ()).throw(_load_failure()),
        )

        outcome = model_ops.benchmark(llama_cfg, config_dir)

        entry = outcome["results"][0]
        assert entry["held_back"] is True and entry["measured"] is False
        assert deletes == [], "benchmark must not delete the user's download"
        record = model_health.read_record(config_dir, "org/m:Q4_K_M")
        assert record["quarantined"] is True
        assert "ran out of memory while loading" in record["quarantine_reason"]

    def test_it_is_not_measured_after_being_held_back(self, env, monkeypatch):
        """Measuring a model that cannot load would just fail again, slowly."""
        llama_cfg, config_dir, _deletes, _freed = env
        monkeypatch.setattr(
            model_ops.model_sizing, "ensure_preset",
            lambda *a, **kw: (_ for _ in ()).throw(_load_failure()),
        )
        measured = []
        monkeypatch.setattr(
            model_ops, "measure", lambda *a, **kw: measured.append(1) or (None, None, None)
        )

        model_ops.benchmark(llama_cfg, config_dir)

        assert measured == []

    def test_a_probe_failure_still_measures_against_the_existing_preset(self, env, monkeypatch):
        llama_cfg, config_dir, _deletes, _freed = env
        monkeypatch.setattr(
            model_ops.model_sizing, "ensure_preset",
            lambda *a, **kw: (_ for _ in ()).throw(_probe_failure()),
        )
        monkeypatch.setattr(model_ops, "measure", lambda *a, **kw: (None, None, "busy"))

        entry = model_ops.benchmark(llama_cfg, config_dir)["results"][0]

        assert not entry.get("held_back")
        assert model_health.read_record(config_dir, "org/m:Q4_K_M") is None
