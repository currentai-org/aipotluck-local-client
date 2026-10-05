"""aipotluck.service.model_health -- surviving a model that runs out of memory at runtime.

The policy these tests pin down is deliberately forgiving, and that is the thing most likely to be
broken by a well-meaning change: an OOM is evidence about the MACHINE AT THAT MOMENT, not about the
model. The machine's free memory moves with whatever else the user is doing, so a context sized on
a quiet box can be too big an hour later. Only a model that keeps failing with no working turn in
between has earned a verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aipotluck.installer import model_presets
from aipotluck.service import model_health as mh


class TestClassifyFailure:
    @pytest.mark.parametrize("line", [
        "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 512.00 MiB on device 0: cudaMalloc failed: out of memory",
        "llama_kv_cache: failed to allocate buffer for kv cache",
        "llama_model_load: unable to allocate CUDA0 buffer",
        "llama_context: failed to allocate compute pp buffers",
        "terminate called after throwing an instance of 'std::bad_alloc'",
        "NvMapMemAllocInternalTagged: error 12",
    ])
    def test_real_allocation_failures_are_recognised(self, line):
        """Every string here comes from the vendored llama.cpp (or, for the last two, from a real
        failure on a Jetson). A signature this misses is an OOM we silently never recover from."""
        assert mh.classify_failure(line, 1) == mh.FAILURE_OOM

    def test_a_signal_kill_is_an_oom_even_with_nothing_logged(self):
        """The OOM killer gives the process no chance to explain itself, so the exit code is the
        only evidence that exists."""
        assert mh.classify_failure("", -9) == mh.FAILURE_OOM
        assert mh.classify_failure("", 137) == mh.FAILURE_OOM

    def test_an_unrelated_failure_is_not_an_oom(self):
        """This is the consequential direction: misreading a corrupt-file failure as an OOM would
        shrink a model's context repeatedly without ever fixing it, then quarantine it for the
        wrong reason."""
        assert mh.classify_failure(
            "llama_model_load: error loading model: invalid magic characters", 1
        ) == mh.FAILURE_OTHER

    def test_a_plain_nonzero_exit_with_a_quiet_log_is_not_assumed_to_be_an_oom(self):
        assert mh.classify_failure("starting up\nshutting down", 1) == mh.FAILURE_OTHER


class TestModelsThatServed:
    """Successes are attributed through the child's PORT, not by proximity in the log, because the
    router interleaves output from every child it proxies for."""

    def test_attributes_a_completed_turn_to_the_right_model(self):
        chunk = "\n".join([
            "33.00.203 I srv proxy_reques: proxying request to model org/alpha:Q4_K_M on port 41673",
            "[41673] 0.26.122 I slot      release: id  0 | task 0 | stop processing: n_tokens = 5317",
        ])
        assert mh.models_that_served(chunk) == {"org/alpha:Q4_K_M"}

    def test_does_not_credit_a_model_that_only_received_a_request(self):
        """Proxying is not serving -- a turn that OOMs mid-flight is proxied too, and crediting it
        would reset the very counter that is supposed to be rising."""
        chunk = "33.00.203 I srv proxy_reques: proxying request to model org/alpha:Q4_K_M on port 41673"
        assert mh.models_that_served(chunk) == set()

    def test_keeps_two_concurrent_models_apart(self):
        """The failure this guards: crediting a success to whichever model was mentioned last."""
        chunk = "\n".join([
            "I srv proxy_reques: proxying request to model org/alpha:Q4_K_M on port 1111",
            "I srv proxy_reques: proxying request to model org/beta:Q8_0 on port 2222",
            "[1111] I slot release: id 0 | task 0 | stop processing: n_tokens = 10",
        ])
        assert mh.models_that_served(chunk) == {"org/alpha:Q4_K_M"}

    def test_ignores_a_success_from_a_port_never_mapped_to_a_model(self):
        chunk = "[9999] I slot release: id 0 | task 0 | stop processing: n_tokens = 10"
        assert mh.models_that_served(chunk) == set()


class TestShrunkContext:
    def test_halves_a_roomy_context(self):
        assert mh.shrunk_context(32768, floor=2048) == 16384

    def test_refuses_to_shrink_below_what_can_hold_a_turn(self):
        """A smaller context is only a recovery while the result can still serve. Past that it is
        a model that loads and cannot converse, which is a worse outcome than saying no."""
        assert mh.shrunk_context(2048, floor=2048) is None
        assert mh.shrunk_context(3000, floor=2048) is None

    def test_halving_rather_than_trimming(self):
        """A 10% trim buys 10% of the headroom that was just proven insufficient, so a model would
        exhaust its retries still failing."""
        assert mh.shrunk_context(8192, floor=2048) == 4096


class TestHealthStore:
    def test_counts_consecutive_ooms(self, tmp_path):
        assert mh.record_oom(tmp_path, "m", from_ctx=32768, to_ctx=16384) == 1
        assert mh.record_oom(tmp_path, "m", from_ctx=16384, to_ctx=8192) == 2

    def test_a_completed_turn_clears_the_count(self, tmp_path):
        """What makes the policy "three OOMs without a working turn" instead of "three OOMs ever".
        Without this, normal use accumulates its way into a quarantine over weeks."""
        mh.record_oom(tmp_path, "m", from_ctx=32768, to_ctx=16384)
        mh.record_success(tmp_path, "m")
        assert mh.read_record(tmp_path, "m")["oom_count"] == 0

    def test_a_success_does_not_resurrect_a_quarantined_model(self, tmp_path):
        mh.record_oom(tmp_path, "m", from_ctx=4096, to_ctx=2048)
        mh.quarantine(tmp_path, "m", "three strikes")
        mh.record_success(tmp_path, "m")
        assert mh.read_record(tmp_path, "m")["quarantined"] is True

    def test_one_models_record_does_not_disturb_another(self, tmp_path):
        mh.record_oom(tmp_path, "a", from_ctx=4096, to_ctx=2048)
        mh.record_oom(tmp_path, "b", from_ctx=8192, to_ctx=4096)
        mh.quarantine(tmp_path, "b", "reason")
        assert mh.read_record(tmp_path, "a")["oom_count"] == 1
        assert not mh.read_record(tmp_path, "a").get("quarantined")

    def test_survives_a_restart(self, tmp_path):
        """The counter has to outlive the process. A model that OOMs hard enough to take the
        service down would otherwise reset its own count on the way back up and retry forever."""
        mh.record_oom(tmp_path, "m", from_ctx=32768, to_ctx=16384)
        mh.record_oom(tmp_path, "m", from_ctx=16384, to_ctx=8192)
        assert mh.read_record(tmp_path, "m")["oom_count"] == 2  # re-read from disk each time

    def test_a_corrupt_store_reads_as_empty_rather_than_raising(self, tmp_path):
        mh.health_path(tmp_path).write_text("{not json", encoding="utf-8")
        assert mh.read_all(tmp_path) == {}

    def test_a_future_schema_is_not_guessed_at(self, tmp_path):
        mh.health_path(tmp_path).write_text(
            json.dumps({"schema_version": 999, "models": {"m": {"oom_count": 9}}}), encoding="utf-8"
        )
        assert mh.read_record(tmp_path, "m") is None

    def test_forget_clears_a_models_history(self, tmp_path):
        mh.record_oom(tmp_path, "m", from_ctx=4096, to_ctx=2048)
        mh.forget(tmp_path, "m")
        assert mh.read_record(tmp_path, "m") is None

    def test_quarantined_models_are_listable_with_their_reason(self, tmp_path):
        mh.quarantine(tmp_path, "m", "ran out of memory 3 times")
        listed = mh.quarantined_models(tmp_path)
        assert listed["m"]["quarantine_reason"] == "ran out of memory 3 times"


def _watcher(tmp_path, monkeypatch, *, models, log_text="", ctx=32768):
    presets = tmp_path / "presets.ini"
    if ctx is not None:
        model_presets.write_preset(presets, "org/m:Q4_K_M", {"ctx-size": str(ctx)})
    log_path = tmp_path / "llama-server.log"
    log_path.write_text(log_text, encoding="utf-8")

    watcher = mh.ModelHealthWatcher(
        base_url="http://127.0.0.1:1", presets_path=presets, config_dir=tmp_path,
        log_path=log_path,
    )
    monkeypatch.setattr(watcher, "_fetch_models", lambda: models)
    reloads = []
    monkeypatch.setattr(watcher, "_request_reload", lambda: reloads.append(1) or True)
    return watcher, presets, reloads


def _failed(model_id="org/m:Q4_K_M", exit_code=1):
    return [{"id": model_id, "status": {"value": "unloaded", "failed": True, "exit_code": exit_code}}]


def _loaded(model_id="org/m:Q4_K_M"):
    return [{"id": model_id, "status": {"value": "loaded"}}]


_OOM_LOG = "[41673] llama_kv_cache: failed to allocate buffer for kv cache\n"


class TestWatcherRecovery:
    def test_an_oom_halves_the_context_and_reloads_the_router(self, tmp_path, monkeypatch):
        watcher, presets, reloads = _watcher(
            tmp_path, monkeypatch, models=_failed(), log_text=_OOM_LOG
        )
        watcher.tick()
        assert model_presets.read_all(presets)["org/m:Q4_K_M"]["ctx-size"] == "16384"
        assert reloads == [1]
        assert mh.read_record(tmp_path, "org/m:Q4_K_M")["oom_count"] == 1

    def test_a_non_oom_failure_leaves_the_context_alone(self, tmp_path, monkeypatch):
        """Shrinking would not fix a corrupt model and would quietly degrade a good one."""
        watcher, presets, reloads = _watcher(
            tmp_path, monkeypatch, models=_failed(),
            log_text="[41673] error loading model: invalid magic characters\n",
        )
        watcher.tick()
        assert model_presets.read_all(presets)["org/m:Q4_K_M"]["ctx-size"] == "32768"
        assert reloads == []
        assert mh.read_record(tmp_path, "org/m:Q4_K_M") is None

    def test_repeated_ooms_quarantine_after_the_limit(self, tmp_path, monkeypatch):
        watcher, presets, _ = _watcher(tmp_path, monkeypatch, models=_failed(), log_text=_OOM_LOG)
        for _ in range(mh.MAX_OOMS_BEFORE_QUARANTINE):
            watcher._last_reacted_at.clear()  # stand in for the cooldown elapsing
            watcher.log_path.write_text(_OOM_LOG, encoding="utf-8")
            watcher._log_offset = 0
            watcher.tick()
        assert mh.read_record(tmp_path, "org/m:Q4_K_M")["quarantined"] is True

    def test_a_turn_between_ooms_prevents_the_quarantine(self, tmp_path, monkeypatch):
        """The whole policy in one test. A user who keeps opening something heavy would otherwise
        lose a model that works perfectly well in between."""
        watcher, _, _ = _watcher(tmp_path, monkeypatch, models=_failed(), log_text=_OOM_LOG)
        success = (
            "I srv proxy_reques: proxying request to model org/m:Q4_K_M on port 41673\n"
            "[41673] I slot release: id 0 | task 0 | stop processing: n_tokens = 10\n"
        )
        for _ in range(5):
            watcher._last_reacted_at.clear()
            watcher.log_path.write_text(_OOM_LOG + success, encoding="utf-8")
            watcher._log_offset = 0
            watcher.tick()
        record = mh.read_record(tmp_path, "org/m:Q4_K_M")
        assert not record.get("quarantined")

    def test_at_the_context_floor_it_waits_rather_than_quarantining(self, tmp_path, monkeypatch):
        """Reaching the floor means no remedy is available this round, not that the model is bad.
        If the squeeze is passing it will serve again and the count clears; if it is not, the
        streak rule below reaches the limit on its own."""
        watcher, presets, _ = _watcher(
            tmp_path, monkeypatch, models=_failed(), log_text=_OOM_LOG, ctx=2048
        )
        watcher.tick()
        record = mh.read_record(tmp_path, "org/m:Q4_K_M")
        assert not record.get("quarantined")
        assert record["oom_count"] == 1
        assert model_presets.read_all(presets)["org/m:Q4_K_M"]["ctx-size"] == "2048"

    def test_a_model_stuck_at_the_floor_is_quarantined_by_the_streak(self, tmp_path, monkeypatch):
        watcher, _, _ = _watcher(
            tmp_path, monkeypatch, models=_failed(), log_text=_OOM_LOG, ctx=2048
        )
        for _ in range(mh.MAX_OOMS_BEFORE_QUARANTINE):
            watcher._last_reacted_at.clear()
            watcher.log_path.write_text(_OOM_LOG, encoding="utf-8")
            watcher._log_offset = 0
            watcher.tick()
        assert mh.read_record(tmp_path, "org/m:Q4_K_M")["quarantined"] is True

    def test_the_cooldown_stops_one_dead_child_being_counted_every_poll(self, tmp_path, monkeypatch):
        """The router keeps reporting `failed` until something loads, so without the cooldown a
        single failure would burn the entire retry budget within seconds."""
        watcher, _, _ = _watcher(tmp_path, monkeypatch, models=_failed(), log_text=_OOM_LOG)
        watcher.tick()
        for _ in range(5):
            watcher.log_path.write_text(_OOM_LOG, encoding="utf-8")
            watcher._log_offset = 0
            watcher.tick()
        assert mh.read_record(tmp_path, "org/m:Q4_K_M")["oom_count"] == 1

    def test_a_quarantined_model_is_left_alone(self, tmp_path, monkeypatch):
        watcher, presets, reloads = _watcher(
            tmp_path, monkeypatch, models=_failed(), log_text=_OOM_LOG
        )
        mh.quarantine(tmp_path, "org/m:Q4_K_M", "already decided")
        watcher.tick()
        assert reloads == []
        assert model_presets.read_all(presets)["org/m:Q4_K_M"]["ctx-size"] == "32768"

    def test_a_healthy_router_changes_nothing(self, tmp_path, monkeypatch):
        watcher, presets, reloads = _watcher(tmp_path, monkeypatch, models=_loaded())
        watcher.tick()
        assert reloads == [] and mh.read_all(tmp_path) == {}

    def test_a_tick_never_raises_when_the_router_is_unreachable(self, tmp_path, monkeypatch):
        """The service's real job works fine without any of this, so the watcher must never be the
        reason it falls over."""
        watcher, _, _ = _watcher(tmp_path, monkeypatch, models=[])
        monkeypatch.undo()
        watcher.tick()  # no router listening on port 1


class TestWatcherLogReading:
    def test_only_reads_what_is_new(self, tmp_path, monkeypatch):
        watcher, _, _ = _watcher(tmp_path, monkeypatch, models=[], log_text="first\n")
        watcher._log_offset = 0
        assert "first" in watcher._read_new_log()
        assert watcher._read_new_log() == ""

    def test_starts_over_when_the_log_is_truncated(self, tmp_path, monkeypatch):
        """Rotation in place. Without this the offset sits past the end of a now-shorter file and
        the watcher goes permanently blind."""
        watcher, _, _ = _watcher(tmp_path, monkeypatch, models=[], log_text="a" * 500 + "\n")
        watcher._read_new_log()
        watcher.log_path.write_text("short\n", encoding="utf-8")
        assert "short" in watcher._read_new_log()

    def test_starts_at_the_end_so_old_history_is_not_replayed(self, tmp_path, monkeypatch):
        """Lines from a previous run were already accounted for; re-reading them would blame models
        twice for the same failures."""
        watcher, _, _ = _watcher(tmp_path, monkeypatch, models=[], log_text=_OOM_LOG * 50)
        watcher._seek_to_end()
        assert watcher._read_new_log() == ""

    def test_a_missing_log_is_not_an_error(self, tmp_path, monkeypatch):
        watcher, _, _ = _watcher(tmp_path, monkeypatch, models=[])
        watcher.log_path = Path(tmp_path / "nope.log")
        assert watcher._read_new_log() == ""
