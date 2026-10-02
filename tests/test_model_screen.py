"""aipotluck.installer.model_screen -- the viability gate that runs before any grading.

Same fake-llama-bench convention as tests/test_model_perf.py: a real executable stand-in whose
timings come from a configurable cost model, so a test can state the device's speed and assert the
verdict that follows from it.
"""

from __future__ import annotations

import json

import pytest

from aipotluck.installer import model_perf as mp
from aipotluck.installer import model_screen as ms
from test_model_perf import fake_bench  # noqa: F401  (shared fixture)


def _budget_s() -> float:
    return mp.TURN_BUDGET_MS / 1000.0


class TestContextFloor:
    def test_a_context_too_small_for_the_system_prompt_is_rejected_without_measuring(
        self, fake_bench, tmp_path, monkeypatch
    ):
        """Free and immediate: no device is fast enough to serve a turn that does not fit."""
        argv_log = tmp_path / "argv.jsonl"
        monkeypatch.setenv("FAKE_BENCH_ARGV_LOG", str(argv_log))

        result = ms.screen_model(fake_bench, "org/repo:Q4_K_M", ctx_size=ms.MIN_CONTEXT_TOKENS - 1)

        assert result.rejected and result.reason_code == ms.REJECT_CONTEXT_TOO_SMALL
        assert "system prompt" in result.reason
        assert not argv_log.exists()  # nothing was benchmarked

    def test_a_sufficient_context_proceeds_to_measurement(self, fake_bench, monkeypatch):
        monkeypatch.setenv("FAKE_BENCH_TG_BASE", "5.0")
        monkeypatch.setenv("FAKE_BENCH_PP_BASE", "0.5")
        result = ms.screen_model(fake_bench, "org/repo:Q4_K_M", ctx_size=32768)
        assert not result.rejected


class TestSpeedFloor:
    def test_a_hopelessly_slow_model_is_rejected_from_the_cheap_probe_alone(
        self, fake_bench, tmp_path, monkeypatch
    ):
        """The point of the screen is a fast "no" -- a model this slow must not cost a full 2k-token
        measurement to reject."""
        argv_log = tmp_path / "argv.jsonl"
        monkeypatch.setenv("FAKE_BENCH_ARGV_LOG", str(argv_log))
        monkeypatch.setenv("FAKE_BENCH_TG_BASE", "2000.0")  # 0.5 tok/s
        monkeypatch.setenv("FAKE_BENCH_PP_BASE", "50.0")

        result = ms.screen_model(fake_bench, "org/repo:Q4_K_M", ctx_size=32768)

        assert result.rejected and result.reason_code == ms.REJECT_TOO_SLOW
        assert result.measured is False
        assert len(argv_log.read_text(encoding="utf-8").splitlines()) == 1

    def test_a_comfortably_fast_model_passes_from_the_cheap_probe_alone(
        self, fake_bench, tmp_path, monkeypatch
    ):
        argv_log = tmp_path / "argv.jsonl"
        monkeypatch.setenv("FAKE_BENCH_ARGV_LOG", str(argv_log))
        monkeypatch.setenv("FAKE_BENCH_TG_BASE", "10.0")   # 100 tok/s
        monkeypatch.setenv("FAKE_BENCH_PP_BASE", "0.5")

        result = ms.screen_model(fake_bench, "org/repo:Q4_K_M", ctx_size=32768)

        assert not result.rejected
        assert result.measured is False
        assert len(argv_log.read_text(encoding="utf-8").splitlines()) == 1

    def test_a_borderline_model_is_measured_rather_than_guessed(self, fake_bench, tmp_path, monkeypatch):
        """The cheap probe extrapolates ~8x, which is fine for a clear case and not fine for a close
        one. Near the line it has to buy the real answer."""
        argv_log = tmp_path / "argv.jsonl"
        monkeypatch.setenv("FAKE_BENCH_ARGV_LOG", str(argv_log))
        # Tuned to land inside the undecided band: ~155s for the 2k-prompt, 100-token turn.
        monkeypatch.setenv("FAKE_BENCH_PP_BASE", "70.0")
        monkeypatch.setenv("FAKE_BENCH_TG_BASE", "100.0")

        result = ms.screen_model(fake_bench, "org/repo:Q4_K_M", ctx_size=32768)

        runs = [json.loads(l) for l in argv_log.read_text(encoding="utf-8").splitlines()]
        assert len(runs) == 2, "a borderline model must be measured, not decided on the probe"
        assert result.measured is True
        # The confirming run is the real contract: a 2k prompt and a 100-token answer.
        last = runs[-1]
        assert last[last.index("-p") + 1] == str(ms.MIN_PROMPT_TOKENS)
        assert last[last.index("-n") + 1] == str(ms.MIN_OUTPUT_TOKENS)


class TestLoadFailure:
    def test_a_model_that_cannot_be_loaded_is_rejected_as_such(self, fake_bench, monkeypatch):
        """Out of memory is its own verdict, not a speed verdict -- the message has to say which."""
        monkeypatch.setenv("FAKE_BENCH_FAIL", "1")
        result = ms.screen_model(fake_bench, "org/repo:Q4_K_M", ctx_size=32768)
        assert result.rejected and result.reason_code == ms.REJECT_WONT_LOAD
        assert "could not load" in result.reason


class TestReporting:
    def test_every_result_carries_its_elapsed_time_and_how_it_was_decided(self, fake_bench, monkeypatch):
        monkeypatch.setenv("FAKE_BENCH_TG_BASE", "10.0")
        monkeypatch.setenv("FAKE_BENCH_PP_BASE", "0.5")
        result = ms.screen_model(fake_bench, "org/repo:Q4_K_M", ctx_size=32768)
        assert result.elapsed_seconds >= 0
        assert result.projected_turn_ms is not None
        assert isinstance(result.measured, bool)
