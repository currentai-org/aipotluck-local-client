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

# Context depths to test the promise at. The grade is stated at the model's FULL context, which for
# a 131k model means a 131k-token prompt -- often too slow to stage here -- so this checks the same
# promise at the depths a run can actually afford and reports which one it reached.
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


def measure_real_turn(base_url: str, model_id: str, words: int, max_tokens: int, *, cold: bool, timeout: float):
    """Runs one real turn and returns (prompt_n, wall_ms). Unloads first by default so the measured
    turn pays the same cold model load the prediction includes -- and the same one a real turn pays,
    since --models-max 1 makes every model switch a cold load."""
    if cold:
        live.unload_model(base_url, model_id)
    started = time.monotonic()
    # A REAL turn of the length the grade claims this model can deliver -- measuring anything
    # shorter would compare a prediction about a full answer against something that was never one.
    timings, _ = live._timed_completion(
        base_url, model_id, words, salt=7, timeout=timeout, max_tokens=max_tokens,
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
    parser.add_argument(
        "--budget-ms", type=float, default=None,
        help="Override the turn budget (default: model_perf.TURN_BUDGET_MS). Every measured turn "
             "is sized to fill this budget, so a smaller value trades coverage for a much faster "
             "run -- and a smaller value is also how you check the grade against a stricter SLA "
             "than the web app's current one.",
    )
    parser.add_argument("--warm", action="store_true",
                        help="Don't unload between turns (measures warm loads; predictions include a cold one)")
    args = parser.parse_args(argv)

    budget_ms = args.budget_ms if args.budget_ms is not None else mp.TURN_BUDGET_MS
    print(f"Measuring {args.model} ... (turn budget {budget_ms/1000:.0f}s)")
    if args.bench_binary:
        result = mp.probe_performance(args.bench_binary, args.model, ctx_size=args.ctx_size)
    else:
        result = live.probe_performance_live(args.base_url, args.model, ctx_size=args.ctx_size)
    fit = result.fit
    safety = mp.SAFETY_FACTOR_LIVE if fit.source == mp.SOURCE_LIVE_SERVER else mp.SAFETY_FACTOR_BENCH
    print(f"  {result.grade.upper()}  n_out={result.n_out:,} tokens  "
          f"{result.decode_tokens_per_second:.1f} tok/s @ {result.ctx_size:,} ctx  "
          f"(output={result.output_grade}, context_cap={result.context_cap}, "
          f"source={fit.source}, confidence={fit.confidence}, safety={safety})")
    print(f"  fit: decode {fit.decode_base_ms:.2f}ms +{fit.decode_depth_ms:.2e}/tok  "
          f"prefill {fit.prefill_base_ms:.3f}ms +{fit.prefill_depth_ms:.2e}/tok  load {fit.load_ms:.0f}ms")

    if result.grade == mp.GRADE_REFUSE:
        print("\nRefused, so there is no promise to check. Nothing to validate.")
        return 0

    depths = [d for d in DEFAULT_GRID if d <= result.ctx_size]
    if args.max_input is not None:
        depths = [d for d in depths if d <= args.max_input] or [min(DEFAULT_GRID)]

    print()
    print("At each depth: how many output tokens the model claims it can deliver, then a real turn")
    print("of exactly that length. The claim holds when the turn lands inside the budget.")
    print()
    print(f"{'depth':>8} {'claimed out':>12} {'predicted':>11} {'actual':>10} {'ratio':>7}  verdict")
    print("-" * 64)

    worst_ratio = 0.0
    overruns = 0
    for depth in depths:
        # What the cost model promises at THIS depth, by the same arithmetic the grade uses.
        prefill_ms = fit.prefill_base_ms * depth + fit.prefill_depth_ms * depth * depth / 2.0
        remaining = budget_ms / safety - fit.load_ms - prefill_ms
        per_token = mp.decode_ms_per_token(fit, depth)
        claimed = int(remaining / per_token) if remaining > 0 and per_token > 0 else 0
        if claimed < 1:
            print(f"{depth:>8} {0:>12} -- nothing claimed at this depth, skipping")
            continue

        try:
            prompt_n, actual_ms = measure_real_turn(
                args.base_url, args.model, depth, claimed, cold=not args.warm, timeout=900,
            )
        except mp.ModelPerfError as exc:
            print(f"{depth:>8}  turn failed: {exc}")
            return 1

        predicted_ms = mp.predict_turn_ms(fit, prompt_n, claimed, decode_depth=depth)
        ratio = actual_ms / predicted_ms if predicted_ms > 0 else float("inf")
        worst_ratio = max(worst_ratio, ratio)
        over = actual_ms > budget_ms * GRID_SLACK
        overruns += int(over)
        verdict = "OVER BUDGET" if over else "ok"
        print(f"{prompt_n:>8} {claimed:>12,} {predicted_ms/1000:>10.1f}s {actual_ms/1000:>9.1f}s "
              f"{ratio:>6.2f}x  {verdict}")

    print()
    print(f"worst under-prediction: {worst_ratio:.2f}x   absorbed by safety factor {safety}: "
          f"{'yes' if worst_ratio <= safety else 'NO'}")

    failures = []
    if overruns:
        failures.append(
            f"{overruns} turn(s) of the length this model claims it can deliver ran past the "
            f"{budget_ms/1000:.0f}s budget -- the grade promises an answer that gets cut off"
        )
    if worst_ratio > safety:
        failures.append(
            f"the cost model under-predicted by {worst_ratio:.2f}x, more than the {safety}x safety "
            "factor absorbs -- raise the factor or improve the fit for this hardware"
        )

    if failures:
        print()
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: the grade's promise holds on this device.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
