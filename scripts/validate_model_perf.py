#!/usr/bin/env python3
"""Check that model_perf's extrapolation actually predicts real turns, on real hardware.

`model_perf` grades a model from a ~2-minute probe and then extrapolates to turns it never ran --
inputs out to 32k tokens with a 1024-token reply. That extrapolation is the load-bearing claim of
the whole feature, and a passing unit suite says nothing about whether it holds: the fake
llama-bench in tests/test_model_perf.py emits timings from the very linear model the fit assumes,
so of course the fit inverts it. Only real hardware can say whether a real model behaves that way.

This script closes that gap. It measures a model, predicts the wall clock for a grid of input
sizes, then ACTUALLY RUNS turns at those sizes against the running router and compares. It is a
gate, not a report: it exits non-zero when the predictions do not hold up, so it can go red for the
thing it is named after.

Run it on every device class we support -- the CPU-only laptop case and the Jetson-class board,
which additionally exercises the no-llama-bench fallback in model_perf_live.

Usage:
    python3 scripts/validate_model_perf.py --model org/repo:Q4_K_M
    python3 scripts/validate_model_perf.py --model org/repo:Q4_K_M --max-input 8192   # quicker
    python3 scripts/validate_model_perf.py --model org/repo:Q4_K_M --base-url http://jetson:8080

Expect this to be slow by construction: each grid point runs a full turn including a cold model
load, and near the top of the grid one turn can approach the web app's whole 155s budget.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aipotluck.installer import model_perf as mp  # noqa: E402
from aipotluck.installer import model_perf_live as live  # noqa: E402

# Spans the turn sizes the web app actually produces: a bare chat turn, a RAG/web-search grounded
# one, and the long-attached-file case that Green is meant to promise is safe.
DEFAULT_GRID = (1024, 4096, 8192, 16384, 32768)

# What this gate is really for. model_perf's grade makes one promise to the user -- a turn with up
# to N_fit tokens of input finishes inside the budget -- so the primary check is to run exactly
# that turn and see. Everything else here is diagnostics.
#
# The secondary check is on the cost model itself. It is known to run optimistic (see
# SAFETY_FACTOR_BENCH's own note), and the safety factor is sized to absorb that; so the thing that
# must hold is that the error stays WITHIN the factor. Once it doesn't, the factor is no longer
# doing its job on this hardware and the grade stops being trustworthy.
TOLERANCE = 0.25          # reported, not gated: how close raw predictions land
GRID_SLACK = 1.10         # a turn measured at N_fit may overshoot the budget by this much before failing


def measure_real_turn(base_url: str, model_id: str, words: int, *, cold: bool, timeout: float):
    """Runs one real turn and returns (prompt_n, wall_ms). Unloads first by default so the measured
    turn pays the same cold model load the prediction includes -- and the same one a real turn pays,
    since --models-max 1 makes every model switch a cold load."""
    if cold:
        live._unload(base_url, model_id)
    started = time.monotonic()
    # A REAL turn, not a probe-shaped one: the web app caps every reply at 1024 tokens, and that is
    # the length predict_turn_ms assumes. Measuring a shorter generation here would compare a
    # prediction about a full turn against something that was never one.
    timings, _ = live._timed_completion(
        base_url, model_id, words, salt=7, timeout=timeout, max_tokens=mp.MAX_OUTPUT_TOKENS,
    )
    wall_ms = (time.monotonic() - started) * 1000.0
    return int(timings["prompt_n"]), wall_ms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face repo[:quant] to validate")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="The running router")
    parser.add_argument("--bench-binary", type=Path, default=None,
                        help="Path to llama-bench; omit to measure through the running server instead")
    parser.add_argument("--ctx-size", type=int, default=131072,
                        help="Context the model is served at (only clamps the grade, not the prediction)")
    parser.add_argument("--max-input", type=int, default=None, help="Skip grid points above this")
    parser.add_argument("--warm", action="store_true",
                        help="Don't unload between turns (measures warm loads; predictions include a cold one)")
    args = parser.parse_args(argv)

    print(f"Measuring {args.model} ...")
    if args.bench_binary:
        result = mp.probe_performance(args.bench_binary, args.model, ctx_size=args.ctx_size)
    else:
        result = live.probe_performance_live(args.base_url, args.model, ctx_size=args.ctx_size)
    fit = result.fit
    safety = mp.SAFETY_FACTOR_LIVE if fit.source == mp.SOURCE_LIVE_SERVER else mp.SAFETY_FACTOR_BENCH
    print(f"  {result.grade.upper()}  n_fit={result.n_fit:,}  {result.decode_tokens_per_second:.1f} tok/s"
          f"  (source={fit.source}, confidence={fit.confidence}, safety={safety})")
    print(f"  fit: decode {fit.decode_base_ms:.2f}ms +{fit.decode_depth_ms:.2e}/tok  "
          f"prefill {fit.prefill_base_ms:.3f}ms +{fit.prefill_depth_ms:.2e}/tok  load {fit.load_ms:.0f}ms")

    if result.grade == mp.GRADE_REFUSE:
        print("\nRefused, so there is no N_fit promise to check. Nothing to validate.")
        return 0

    # The grid exists for diagnostics; N_fit is the point the promise is actually about.
    grid = sorted({n for n in DEFAULT_GRID if n <= result.n_fit} | {result.n_fit})
    if args.max_input is not None:
        grid = [n for n in grid if n <= args.max_input] or [min(grid)]

    print()
    print(f"{'input':>8} {'predicted':>11} {'actual':>10} {'error':>9} {'ratio':>7}  note")
    print("-" * 62)

    worst_ratio = 0.0
    n_fit_actual_ms = None
    for target in grid:
        try:
            prompt_n, actual_ms = measure_real_turn(
                args.base_url, args.model, target, cold=not args.warm, timeout=600,
            )
        except mp.ModelPerfError as exc:
            print(f"{target:>8}  turn failed: {exc}")
            return 1

        # Predict at the token count the server actually saw, so tokenizer drift in the filler
        # isn't scored as extrapolation error.
        predicted_ms = mp.predict_turn_ms(fit, prompt_n)
        error = (predicted_ms - actual_ms) / actual_ms
        ratio = actual_ms / predicted_ms if predicted_ms > 0 else float("inf")
        worst_ratio = max(worst_ratio, ratio)
        note = "<- N_fit" if target == result.n_fit else ""
        if target == result.n_fit:
            n_fit_actual_ms = actual_ms
        print(f"{prompt_n:>8} {predicted_ms/1000:>10.1f}s {actual_ms/1000:>9.1f}s "
              f"{error*100:>+8.1f}% {ratio:>6.2f}x  {note}")

    print()
    print(f"worst under-prediction: {worst_ratio:.2f}x   absorbed by safety factor {safety}: "
          f"{'yes' if worst_ratio <= safety else 'NO'}")

    failures = []
    if n_fit_actual_ms is not None and n_fit_actual_ms > mp.TURN_BUDGET_MS * GRID_SLACK:
        failures.append(
            f"a real turn at N_fit={result.n_fit:,} took {n_fit_actual_ms/1000:.0f}s, over the "
            f"{mp.TURN_BUDGET_MS/1000:.0f}s budget -- the grade promises a turn that times out"
        )
    if worst_ratio > safety:
        failures.append(
            f"the cost model under-predicted by {worst_ratio:.2f}x, more than the {safety}x safety "
            "factor absorbs -- raise the factor or improve the fit for this hardware"
        )
    if result.grade == mp.GRADE_GREEN and n_fit_actual_ms and n_fit_actual_ms > mp.TURN_BUDGET_MS:
        failures.append("a GREEN model exceeded the budget -- green is supposed to mean this cannot happen")

    if failures:
        print()
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: the grade's promise holds on this device.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
