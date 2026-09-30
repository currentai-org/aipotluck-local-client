"""aipotluck.installer.model_perf -- the cost model, the fit, and the llama-bench probe.

Boundary convention follows tests/test_model_sizing.py: a REAL executable stand-in for the
external binary, written to tmp_path, rather than a mock of subprocess. The fake llama-bench below
computes its timings from an internally consistent linear cost model, so a test can configure a
known `base + slope*depth` and assert the two-point fit recovers exactly those coefficients --
which checks that the fit actually inverts the model rather than merely producing some number.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from aipotluck.installer import model_perf as mp

FAKE_BENCH_SCRIPT = r'''#!/usr/bin/env python3
import json, os, sys

args = sys.argv[1:]
log_path = os.environ.get("FAKE_BENCH_ARGV_LOG")
if log_path:
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(args) + "\n")

if os.environ.get("FAKE_BENCH_FAIL"):
    sys.stderr.write("fake llama-bench refused\n")
    sys.exit(2)
if os.environ.get("FAKE_BENCH_GARBAGE"):
    sys.stdout.write("not json at all\n")
    sys.exit(0)


def flag(name, default):
    return type(default)(args[args.index(name) + 1]) if name in args else default


depth = flag("-d", 0)
n_prompt = flag("-p", 512)
n_gen = flag("-n", 128)
reps = flag("-r", 5)

pp_base = float(os.environ.get("FAKE_BENCH_PP_BASE", "1.0"))
pp_slope = float(os.environ.get("FAKE_BENCH_PP_SLOPE", "0.0"))
tg_base = float(os.environ.get("FAKE_BENCH_TG_BASE", "20.0"))
tg_slope = float(os.environ.get("FAKE_BENCH_TG_SLOPE", "0.0"))
noise = float(os.environ.get("FAKE_BENCH_NOISE", "0.0"))

pp_cost = pp_base + pp_slope * (depth + n_prompt / 2.0)
tg_cost = tg_base + tg_slope * (depth + n_gen / 2.0)


def samples(per_token_ms, count):
    # The LAST repetition is the slowest, so a test can tell "took the max" apart from "took the
    # mean" or "took the first".
    return [int(per_token_ms * count * 1e6 * (1.0 + noise * i)) for i in range(reps)]


common = {
    "model_filename": os.environ.get("FAKE_BENCH_MODEL_PATH", "/tmp/fake-model.gguf"),
    "model_size": int(os.environ.get("FAKE_BENCH_MODEL_SIZE", "1048576")),
    "n_depth": depth,
}
out = [
    dict(common, n_prompt=n_prompt, n_gen=0, samples_ns=samples(pp_cost, n_prompt)),
    dict(common, n_prompt=0, n_gen=n_gen, samples_ns=samples(tg_cost, n_gen)),
]
sys.stdout.write("load_backend: chatter that precedes the JSON\n")
sys.stdout.write(json.dumps(out))
sys.exit(0)
'''


@pytest.fixture
def fake_bench(tmp_path: Path) -> Path:
    script = tmp_path / "llama-bench"
    script.write_text(FAKE_BENCH_SCRIPT, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


def make_fit(tg_tps=50.0, pp_tps=400.0, tg_slope=0.0, pp_slope=0.0, load_ms=0.0,
             confidence=mp.CONFIDENCE_OK, source=mp.SOURCE_LLAMA_BENCH) -> mp.PerfFit:
    return mp.PerfFit(
        decode_base_ms=1000.0 / tg_tps, decode_depth_ms=tg_slope,
        prefill_base_ms=1000.0 / pp_tps, prefill_depth_ms=pp_slope,
        load_ms=load_ms, source=source, confidence=confidence,
    )


class TestSolveNFit:
    def test_returns_zero_when_the_reply_alone_exceeds_the_budget(self):
        """The output cap is 1024 tokens, so a slow enough model times out no matter how short the
        prompt is. That has to read as a refusal, not as a small but workable input budget."""
        fit = make_fit(tg_tps=5.0)  # 1024 tokens at 5 tok/s = 204.8s, over any 155s budget
        assert mp.solve_n_fit(fit, mp.TURN_BUDGET_MS) == 0

    def test_a_turn_at_n_fit_lands_inside_the_budget_and_one_token_more_does_not(self):
        fit = make_fit(tg_tps=60.0, pp_tps=900.0, tg_slope=6e-5, pp_slope=2e-6, load_ms=3_000)
        budget = 120_000.0
        n = mp.solve_n_fit(fit, budget)
        assert mp.predict_turn_ms(fit, n) <= budget
        assert mp.predict_turn_ms(fit, n + 1) > budget

    def test_load_time_eats_into_the_input_budget(self):
        fast = make_fit(tg_tps=60.0, pp_tps=900.0, tg_slope=6e-5)
        slow_disk = make_fit(tg_tps=60.0, pp_tps=900.0, tg_slope=6e-5, load_ms=60_000)
        assert mp.solve_n_fit(slow_disk, 120_000.0) < mp.solve_n_fit(fast, 120_000.0)


class TestComputeGrade:
    @pytest.mark.parametrize(
        "n_fit,expected",
        [
            (mp.GREEN_MIN_TOKENS, mp.GRADE_GREEN),
            (mp.GREEN_MIN_TOKENS - 1, mp.GRADE_YELLOW),
            (mp.YELLOW_MIN_TOKENS, mp.GRADE_YELLOW),
            (mp.YELLOW_MIN_TOKENS - 1, mp.GRADE_RED),
            (mp.RED_MIN_TOKENS, mp.GRADE_RED),
            (mp.RED_MIN_TOKENS - 1, mp.GRADE_REFUSE),
            (0, mp.GRADE_REFUSE),
        ],
    )
    def test_grade_boundaries(self, n_fit, expected):
        assert mp.grade_for(n_fit) == expected

    def test_context_window_clamps_the_answer_below_what_speed_allows(self):
        """The web app never truncates an over-long prompt, so a turn that doesn't fit the context
        window fails outright however fast the model is -- the grade has to reflect that."""
        fit = make_fit(tg_tps=200.0, pp_tps=5000.0)
        result = mp.compute_grade(fit, ctx_size=8192)
        assert result.n_fit == 8192 - mp.MAX_OUTPUT_TOKENS
        assert result.grade == mp.GRADE_RED
        assert "context" in result.reason

    def test_a_small_context_model_cannot_be_green_at_any_speed(self):
        result = mp.compute_grade(make_fit(tg_tps=1000.0, pp_tps=50_000.0), ctx_size=4096)
        assert result.grade != mp.GRADE_GREEN

    def test_low_confidence_cannot_certify_green(self):
        """Green is a promise about heavy use; a noisy or depth-less measurement can't support one."""
        fast = make_fit(tg_tps=200.0, pp_tps=5000.0)
        assert mp.compute_grade(fast, ctx_size=131072).grade == mp.GRADE_GREEN
        unsure = make_fit(tg_tps=200.0, pp_tps=5000.0, confidence=mp.CONFIDENCE_LOW)
        graded = mp.compute_grade(unsure, ctx_size=131072)
        assert graded.grade == mp.GRADE_YELLOW
        assert "noisy" in graded.reason or "short" in graded.reason

    def test_low_confidence_does_not_rescue_a_refusal(self):
        unsure = make_fit(tg_tps=4.0, confidence=mp.CONFIDENCE_LOW)
        assert mp.compute_grade(unsure, ctx_size=131072).grade == mp.GRADE_REFUSE

    def test_the_live_server_path_is_graded_more_conservatively(self):
        """Same measured speed, noisier measurement path -> a smaller promised input budget."""
        bench = mp.compute_grade(make_fit(tg_tps=30.0, source=mp.SOURCE_LLAMA_BENCH), ctx_size=131072)
        live = mp.compute_grade(make_fit(tg_tps=30.0, source=mp.SOURCE_LIVE_SERVER), ctx_size=131072)
        assert live.n_fit < bench.n_fit

    def test_never_raises_on_degenerate_input(self):
        degenerate = mp.PerfFit(0.0, 0.0, 0.0, 0.0, 0.0, mp.SOURCE_LLAMA_BENCH, mp.CONFIDENCE_OK)
        assert mp.compute_grade(degenerate, ctx_size=0).n_fit == 0


class TestFitLine:
    def test_recovers_a_known_line(self):
        base, slope = mp._fit_line(100.0, 12.0, 4100.0, 20.0)
        assert slope == pytest.approx(0.002)
        assert base == pytest.approx(11.8)

    def test_a_negative_slope_is_flattened_and_the_slower_cost_kept(self):
        """Noise can make the deeper measurement look faster. Taken at face value that extrapolates
        to a model that gets cheaper with more context, which is unboundedly optimistic -- it would
        hand back an enormous N_fit and a green grade for a model that deserves neither."""
        base, slope = mp._fit_line(100.0, 20.0, 4100.0, 18.0)
        assert slope == 0.0
        assert base == 20.0

    def test_a_single_point_leaves_the_slopes_flat_and_the_confidence_low(self):
        point = mp.BenchPoint(0, 1.0, 20.0, 128.0, 16.0, noisy=False)
        fit = mp.fit_points(point, None, 0.0, mp.SOURCE_LLAMA_BENCH)
        assert (fit.prefill_depth_ms, fit.decode_depth_ms) == (0.0, 0.0)
        assert fit.confidence == mp.CONFIDENCE_LOW


class TestRunBenchPoint:
    def test_benchmarks_at_the_presets_operating_point(self, fake_bench, tmp_path, monkeypatch):
        """The preset's K/V cache types and GPU-layer count must reach llama-bench. Measuring at
        llama-bench's own defaults instead would be a silent systematic error -- a q4_0 K/V cache
        materially changes decode speed -- and it would be invisible in the resulting number."""
        argv_log = tmp_path / "argv.jsonl"
        monkeypatch.setenv("FAKE_BENCH_ARGV_LOG", str(argv_log))

        mp.run_bench_point(
            fake_bench, "org/repo:Q4_K_M", depth=2048,
            cache_type_k="q8_0", cache_type_v="q4_0", gpu_layers=33, timeout=30,
        )

        args = json.loads(argv_log.read_text(encoding="utf-8").splitlines()[0])
        assert args[args.index("-ctk") + 1] == "q8_0"
        assert args[args.index("-ctv") + 1] == "q4_0"
        assert args[args.index("-ngl") + 1] == "33"
        assert args[args.index("-d") + 1] == "2048"
        assert args[args.index("-hf") + 1] == "org/repo:Q4_K_M"
        # --offline: the model was just downloaded, so a cache miss here means something is wrong
        # and should fail rather than quietly re-fetching gigabytes.
        assert "--offline" in args
        # Thread count is deliberately absent -- llama-bench defaults to the same
        # common_cpu_get_num_math() llama-server does, so passing a guess would MIS-match it.
        assert "-t" not in args

    def test_non_numeric_gpu_layers_is_omitted_rather_than_passed_through(self, fake_bench, tmp_path, monkeypatch):
        """llama-server accepts --gpu-layers auto/all; llama-bench wants a count and would reject
        the word, failing the probe for a setting that is perfectly valid upstream."""
        argv_log = tmp_path / "argv.jsonl"
        monkeypatch.setenv("FAKE_BENCH_ARGV_LOG", str(argv_log))
        mp.run_bench_point(fake_bench, "org/repo:Q4_K_M", depth=0, gpu_layers="auto", timeout=30)
        assert "-ngl" not in json.loads(argv_log.read_text(encoding="utf-8").splitlines()[0])

    def test_uses_the_slowest_repetition_not_the_mean(self, fake_bench, monkeypatch):
        monkeypatch.setenv("FAKE_BENCH_TG_BASE", "20.0")
        monkeypatch.setenv("FAKE_BENCH_NOISE", "0.5")  # repetitions at 1.0x, 1.5x, 2.0x of 20ms
        point, _ = mp.run_bench_point(fake_bench, "org/repo:Q4_K_M", depth=0, timeout=30)
        assert point.decode_ms_per_token == pytest.approx(40.0)

    def test_a_high_spread_across_repetitions_is_flagged_noisy(self, fake_bench, monkeypatch):
        monkeypatch.setenv("FAKE_BENCH_NOISE", "0.5")
        point, _ = mp.run_bench_point(fake_bench, "org/repo:Q4_K_M", depth=0, timeout=30)
        assert point.noisy is True

    def test_jitter_on_a_very_fast_run_is_not_called_noise(self, fake_bench, monkeypatch):
        """Measured on a Jetson Orin NX: a GPU-served 0.5B model's whole probe ran in 7.2s, and
        ordinary scheduler jitter across sub-second repetitions cleared 20% easily. Treating that
        as low confidence would cap genuinely fast models at yellow over tens of milliseconds of
        noise against a 155,000ms budget."""
        monkeypatch.setenv("FAKE_BENCH_TG_BASE", "0.2")  # 32 tokens -> ~6ms a repetition
        monkeypatch.setenv("FAKE_BENCH_PP_BASE", "0.01")
        monkeypatch.setenv("FAKE_BENCH_NOISE", "0.5")  # 50% relative spread, a few ms absolute
        point, _ = mp.run_bench_point(fake_bench, "org/repo:Q4_K_M", depth=0, timeout=30)
        assert point.noisy is False

    def test_a_failing_binary_raises_rather_than_returning_a_guess(self, fake_bench, monkeypatch):
        monkeypatch.setenv("FAKE_BENCH_FAIL", "1")
        with pytest.raises(mp.ModelPerfError, match="exited 2"):
            mp.run_bench_point(fake_bench, "org/repo:Q4_K_M", depth=0, timeout=30)

    def test_unparseable_output_raises(self, fake_bench, monkeypatch):
        monkeypatch.setenv("FAKE_BENCH_GARBAGE", "1")
        with pytest.raises(mp.ModelPerfError, match="no JSON"):
            mp.run_bench_point(fake_bench, "org/repo:Q4_K_M", depth=0, timeout=30)


class TestProbePerformance:
    def test_two_points_recover_the_cost_model_they_were_generated_from(self, fake_bench, monkeypatch):
        """The whole design rests on a two-point linear-in-depth fit inverting the real cost curve.
        The fake emits timings from an exactly linear model, so the fitted coefficients must come
        back as the ones it was configured with."""
        monkeypatch.setenv("FAKE_BENCH_TG_BASE", "20.0")
        monkeypatch.setenv("FAKE_BENCH_TG_SLOPE", "0.001")
        monkeypatch.setenv("FAKE_BENCH_PP_BASE", "2.0")
        monkeypatch.setenv("FAKE_BENCH_PP_SLOPE", "0.0002")

        result = mp.probe_performance(fake_bench, "org/repo:Q4_K_M", ctx_size=131072, budget_seconds=600)

        assert result.fit.decode_base_ms == pytest.approx(20.0, rel=1e-6)
        assert result.fit.decode_depth_ms == pytest.approx(0.001, rel=1e-6)
        assert result.fit.prefill_base_ms == pytest.approx(2.0, rel=1e-6)
        assert result.fit.prefill_depth_ms == pytest.approx(0.0002, rel=1e-6)
        # The quoted rate is taken at a realistic context depth, not at the fit's depth-0 intercept
        # (which would be 50 tok/s here) -- no real turn runs at depth 0, and quoting the intercept
        # would flatter every model by exactly the amount its context costs it.
        expected = 1000.0 / (20.0 + 0.001 * mp._REPORTED_RATE_DEPTH)
        assert result.decode_tokens_per_second == pytest.approx(expected, rel=1e-6)
        assert result.decode_tokens_per_second < 50.0

    def test_a_hopeless_model_is_refused_without_paying_for_a_deep_probe(self, fake_bench, tmp_path, monkeypatch):
        """A model that can't emit 1024 tokens at zero context won't be saved by anything a deeper
        measurement could show, and the user is already waiting -- so stop after one run."""
        argv_log = tmp_path / "argv.jsonl"
        monkeypatch.setenv("FAKE_BENCH_ARGV_LOG", str(argv_log))
        monkeypatch.setenv("FAKE_BENCH_TG_BASE", "300.0")  # ~3.3 tok/s

        result = mp.probe_performance(fake_bench, "org/repo:Q4_K_M", ctx_size=131072, budget_seconds=600)

        assert result.grade == mp.GRADE_REFUSE
        assert result.n_fit == 0
        assert len(argv_log.read_text(encoding="utf-8").splitlines()) == 1

    def test_a_viable_model_gets_a_second_deeper_point(self, fake_bench, tmp_path, monkeypatch):
        argv_log = tmp_path / "argv.jsonl"
        monkeypatch.setenv("FAKE_BENCH_ARGV_LOG", str(argv_log))
        monkeypatch.setenv("FAKE_BENCH_TG_BASE", "20.0")

        mp.probe_performance(fake_bench, "org/repo:Q4_K_M", ctx_size=131072, budget_seconds=600)

        runs = [json.loads(line) for line in argv_log.read_text(encoding="utf-8").splitlines()]
        assert len(runs) == 2
        assert runs[0][runs[0].index("-d") + 1] == str(mp._SHALLOW_DEPTH)
        assert runs[1][runs[1].index("-d") + 1] == str(mp._CANDIDATE_DEPTHS[0])

    def test_the_shallow_anchor_is_not_depth_zero(self):
        """Measured on a real CPU-only laptop: decode cost per token was 17.3ms at depth 0 but
        32.8ms at 512 and 33.9ms at 1024 -- it nearly doubles over the first few hundred tokens,
        because generating at depth 0 does almost no attention work. Fitting a straight line
        through that point made the whole model OPTIMISTIC across the range the app really uses
        (-26% at 512, -9% at 1024), which is the direction that promises turns that then time out.
        The app never sends a near-empty prompt anyway -- its system prompt alone is ~1,450 tokens.
        """
        assert mp._SHALLOW_DEPTH > 0
        assert all(depth > mp._SHALLOW_DEPTH for depth in mp._CANDIDATE_DEPTHS)

    def test_a_slow_device_falls_back_to_a_shallower_second_point(self, fake_bench, tmp_path, monkeypatch):
        """Prefilling to 4096 costs real time on slow hardware. Rather than blow the probe budget,
        settle for a shorter lever arm -- the verdict there is dominated by the depth-0 term."""
        argv_log = tmp_path / "argv.jsonl"
        monkeypatch.setenv("FAKE_BENCH_ARGV_LOG", str(argv_log))
        monkeypatch.setenv("FAKE_BENCH_TG_BASE", "60.0")
        monkeypatch.setenv("FAKE_BENCH_PP_BASE", "12.0")  # 4096 tokens of prefill would take ~49s a pass

        mp.probe_performance(fake_bench, "org/repo:Q4_K_M", ctx_size=131072, budget_seconds=120)

        runs = [json.loads(line) for line in argv_log.read_text(encoding="utf-8").splitlines()]
        assert len(runs) == 2
        assert int(runs[1][runs[1].index("-d") + 1]) < mp._CANDIDATE_DEPTHS[0]

    def test_a_missing_binary_raises(self, tmp_path):
        with pytest.raises(mp.ModelPerfError, match="not found"):
            mp.probe_performance(tmp_path / "nope", "org/repo:Q4_K_M", ctx_size=4096)


class TestMeasureColdLoadMs:
    def test_scales_a_sampled_read_rate_up_to_the_whole_file(self, tmp_path):
        blob = tmp_path / "model.gguf"
        blob.write_bytes(b"\0" * (4 * 1024 * 1024))
        measured = mp.measure_cold_load_ms(blob, model_size_bytes=64 * 1024 * 1024)
        if measured is None:
            pytest.skip("posix_fadvise is unavailable on this platform")
        assert measured > 0

    def test_returns_none_for_an_unreadable_file(self, tmp_path):
        assert mp.measure_cold_load_ms(tmp_path / "missing.gguf", model_size_bytes=1024) is None
