"""The reject screen: does this model have any business being installed at all?

Grading says how good a model is. This says whether it is viable, and it runs FIRST, because the
answer is usually obvious and a user who is about to be told "no" should not wait out a full
benchmark to hear it. Only models that survive this get measured properly.

Three ways to fail, all worst-case rather than preference:

1. **It cannot be loaded.** Out of memory on this device, or a file llama.cpp refuses. Nothing to
   grade; nothing a faster machine-state would fix.
2. **Its context is under 2k tokens.** The web app's system prompt alone is roughly 1,450 tokens
   before any conversation, retrieval or attachment is added, so a smaller window cannot hold a
   single turn.
3. **It cannot produce 100 output tokens from a 2k-token prompt inside the 155s budget.** That is
   the shortest useful exchange the app ever asks for -- a brief, direct answer to a short question
   with the system prompt in front of it. A model that misses this does not have a slow mode, it
   has no working mode.

## Why it is staged

The expensive confirmation is a real 2048-token prefill followed by 100 generated tokens. On slow
hardware that measurement can itself approach the budget it is testing, which is useless for a
screen meant to answer quickly. So a cheap shallow probe goes first and settles the clear cases --
hopeless or comfortable -- and only a genuinely borderline model pays for the real thing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from aipotluck.installer import model_perf

log = logging.getLogger("aipotluck.installer.model_screen")

# The shortest exchange the app ever asks for: its system prompt plus a short question, answered
# briefly. Both numbers are the user-facing contract this screen enforces.
MIN_PROMPT_TOKENS = 2_048
MIN_OUTPUT_TOKENS = 100

# A context smaller than this cannot hold the system prompt, so no turn is servable at any speed.
MIN_CONTEXT_TOKENS = 2_048

# The cheap probe extrapolates ~8x, so it only decides a case that is clear by a wide margin.
# Inside this band the verdict is bought with a real measurement instead of guessed.
_DECISIVE_MARGIN = 1.5

# Shape of the cheap probe. The prompt length is deliberately 1024 rather than something smaller:
# prefill cost per token is NOT flat across prompt lengths, and a short probe lands in a different
# regime from the 2048-token prompt this screen is about. Measured on an i7 laptop, prefill is
# 4.33 ms/token at a 384-token prompt and 10.05 at 2048 -- llama.cpp's default n_ubatch is 512 and
# the step sits right there. Projecting the 2048 case from a 256-token probe therefore understated
# it by about 2.3x, in the optimistic direction, which is the wrong way for a gate to be wrong.
# 1024 is on the far side of that step (9.37 ms/token) and costs half as much as measuring 2048.
_PROBE_PROMPT = 1024
_PROBE_GEN = 16
_PROBE_DEPTH = 1024

REJECT_WONT_LOAD = "wont_load"
REJECT_CONTEXT_TOO_SMALL = "context_too_small"
REJECT_TOO_SLOW = "too_slow"


@dataclass
class ScreenResult:
    rejected: bool
    reason_code: str | None
    reason: str
    projected_turn_ms: float | None  # the 2k-prompt, 100-token turn this screen is about
    measured: bool                   # True when a real turn was run, False when projected
    elapsed_seconds: float


def _ok(detail: str, projected: float | None, measured: bool, started: float) -> ScreenResult:
    return ScreenResult(False, None, detail, projected, measured, time.monotonic() - started)


def _reject(code: str, detail: str, projected: float | None, measured: bool, started: float) -> ScreenResult:
    return ScreenResult(True, code, detail, projected, measured, time.monotonic() - started)


def screen_model(
    bench_binary: Path,
    model_id: str,
    *,
    ctx_size: int,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    gpu_layers: str | int | None = None,
    budget_seconds: float = 60.0,
) -> ScreenResult:
    """Decides whether `model_id` is viable at all. Cheap checks first, and a real measurement only
    where the cheap one cannot honestly call it."""
    started = time.monotonic()

    if ctx_size < MIN_CONTEXT_TOKENS:
        return _reject(
            REJECT_CONTEXT_TOO_SMALL,
            f"its {ctx_size:,}-token context cannot hold the app's system prompt, which is about "
            f"1,450 tokens before anything is added",
            None, False, started,
        )

    def bench(args_prompt: int, args_gen: int, depth: int, timeout: float):
        return model_perf.run_bench_point(
            bench_binary, model_id, depth=depth,
            cache_type_k=cache_type_k, cache_type_v=cache_type_v, gpu_layers=gpu_layers,
            timeout=timeout, n_prompt=args_prompt, n_gen=args_gen,
        )

    # --- cheap probe: settles anything that is not close ---
    try:
        point, _ = bench(_PROBE_PROMPT, _PROBE_GEN, _PROBE_DEPTH, timeout=budget_seconds)
    except model_perf.ModelPerfError as exc:
        # llama-bench refusing to load is the out-of-memory case, and it is a real verdict.
        if "exited" in str(exc) or "not found" in str(exc):
            return _reject(
                REJECT_WONT_LOAD,
                f"this device could not load it at all ({str(exc).splitlines()[0][:160]})",
                None, False, started,
            )
        raise

    projected = (
        point.prefill_ms_per_token * MIN_PROMPT_TOKENS
        + point.decode_ms_per_token * MIN_OUTPUT_TOKENS
    )
    budget = model_perf.TURN_BUDGET_MS
    if projected > budget * _DECISIVE_MARGIN:
        return _reject(
            REJECT_TOO_SLOW,
            f"a {MIN_PROMPT_TOKENS:,}-token question answered in {MIN_OUTPUT_TOKENS} tokens needs "
            f"about {projected/1000:.0f}s here, well past the {budget/1000:.0f}s budget",
            projected, False, started,
        )
    if projected * _DECISIVE_MARGIN < budget:
        return _ok(
            f"comfortably serves the shortest turn (about {projected/1000:.0f}s against a "
            f"{budget/1000:.0f}s budget)",
            projected, False, started,
        )

    # --- borderline: buy the real answer rather than guess it ---
    log.info("%s is borderline on the quick screen -- measuring a real short turn", model_id)
    remaining = budget_seconds - (time.monotonic() - started)
    try:
        real, _ = bench(MIN_PROMPT_TOKENS, MIN_OUTPUT_TOKENS, 0,
                        timeout=max(5.0, min(remaining, budget / 1000.0)))
    except model_perf.ModelPerfError as exc:
        # Running out of time IS the finding: the turn did not fit the budget it was given.
        return _reject(
            REJECT_TOO_SLOW,
            f"a {MIN_PROMPT_TOKENS:,}-token question answered in {MIN_OUTPUT_TOKENS} tokens did not "
            f"finish in the time available ({exc})",
            projected, True, started,
        )

    measured_ms = (
        real.prefill_ms_per_token * MIN_PROMPT_TOKENS + real.decode_ms_per_token * MIN_OUTPUT_TOKENS
    )
    if measured_ms > budget:
        return _reject(
            REJECT_TOO_SLOW,
            f"a {MIN_PROMPT_TOKENS:,}-token question answered in {MIN_OUTPUT_TOKENS} tokens really "
            f"takes {measured_ms/1000:.0f}s here, past the {budget/1000:.0f}s budget",
            measured_ms, True, started,
        )
    return _ok(
        f"serves the shortest turn in about {measured_ms/1000:.0f}s against a {budget/1000:.0f}s budget",
        measured_ms, True, started,
    )
