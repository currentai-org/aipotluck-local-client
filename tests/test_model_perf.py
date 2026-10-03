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


class TestWaitUntilIdle:
    """Bailing the moment load looks high is wrong when the load is OURS. Benchmarking several
    models re-sizes each one first -- a full model load apiece -- so the one-minute average still
    carries the previous model's probe. Measured on a Jetson: the first model graded, then every
    later one was skipped as "busy" by work this same command had just finished."""

    def test_returns_immediately_when_already_idle(self, monkeypatch):
        monkeypatch.setattr(mp, "machine_is_too_busy", lambda: False)
        assert mp.wait_until_idle(timeout_seconds=0.0) is True

    def test_waits_for_transient_load_to_clear(self, monkeypatch):
        readings = iter([True, True, False])
        monkeypatch.setattr(mp, "machine_is_too_busy", lambda: next(readings))
        monkeypatch.setattr(mp.time, "sleep", lambda _s: None)
        assert mp.wait_until_idle(timeout_seconds=60.0) is True

    def test_gives_up_on_a_machine_that_stays_busy(self, monkeypatch):
        """A machine genuinely occupied by something else still gets declined -- the point is to
        wait out our own noise, not to measure through someone else's workload."""
        monkeypatch.setattr(mp, "machine_is_too_busy", lambda: True)
        monkeypatch.setattr(mp.time, "sleep", lambda _s: None)
        assert mp.wait_until_idle(timeout_seconds=0.01) is False


