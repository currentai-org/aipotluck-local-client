"""Per-model performance grading: will this model actually finish a turn before the web app
gives up on it? (CUR-TBD)

On a small device -- a CPU-only mini PC, a Jetson-class board -- it is very easy to `pull` a model
that loads fine, appears in the chat model picker, and then simply never finishes a reply. This
module measures the real thing on the real hardware at pull time and turns it into one grade plus
one comparable number, so that failure is caught at install time instead of mid-conversation.

## The budget this grades against

`aipotluck.org`'s `web/src/lib/chat/utils/deadlineLadder.ts:255` derives
`MAX_SAFE_GENERATION_TIMEOUT_MS = 180k - 15k - 2*5k = 155_000`, and a local model's turn is clamped
to exactly that (`web/src/routes/chat/conversation/[id]/+server.ts:1164-1178`). The clock starts
*after* that app's prestream chain, so the budget covers **cold model load + prefill + decode** --
all three are ours to pay. The often-quoted "160 seconds" appears nowhere in that repo; it is
folklore, and it is 5s optimistic, so this module uses the derived number.

Two more facts from that side set the shape of the estimate:

- **Output is hard-capped at 1024 tokens** (`GROUNDED_DECODING.max_tokens`, merged last over any
  model config). So decode alone can blow the budget *regardless of how short the prompt is* --
  1024 tokens at 5 tok/s is 205s. That case is checked first and short-circuits the whole probe.
- **Input is uncapped and never truncated** (the model config's `truncate` field has no consumers
  and local models declare no context length at all). A typical turn is ~2k tokens, 8-10k once RAG
  or web-search grounding fires, 32k+ with an attached file. A turn that overflows the model's
  context window fails outright rather than degrading, which is why `ctx_size` clamps the answer
  below rather than merely informing it.

## How the estimate works

Standard decomposition (`E2E = TTFT + (n_out - 1) * TPOT`), but fitted on per-token *time* rather
than on rates. Decode re-reads a KV cache that grows linearly with depth, so the cost of one token
at depth d is `a + b*d`; that closed form matters here because this package is stdlib-only
(`pyproject.toml`: `dependencies = []`) and there is no numeric integrator to reach for:

    prefill token cost at depth d:  c + e*d   ->  T_prefill(n) = c*n + e*n^2/2
    decode  token cost at depth d:  a + b*d

`llama-bench -d <depth>` prefills the KV cache to a given depth before measuring, which is exactly
the measurement this needs, and two points are enough to fit both lines.

**The metric is `n_out`**: how many tokens the model can generate before the stream is cut off,
with generation priced at the model's full context depth -- the slowest it will ever run. That is
the failure users actually feel; a reply that stops mid-thought is the problem, and prefill is
cheap next to generation. Grading on how large an INPUT fits made the verdict mostly a restatement
of the context size, which is not the interesting question.

Context has not stopped mattering, it just enters twice and in its proper place: it slows every
generated token (through `b * ctx_size`), and it caps the grade outright (`context_grade_cap`),
because a model that generates quickly is still limited by how much it can be told.

## Why the benchmark must copy the preset

`llama-bench` is run with the K/V cache types and GPU-layer count that `model_sizing.compute_sizing`
just wrote into this model's `--models-preset` section. Benchmarking at llama-bench's own defaults
instead would be a silent systematic error larger than the fit error, because a q4_0 K/V cache
materially changes decode speed. Thread count is deliberately *not* passed: llama-bench's default
is `common_cpu_get_num_math()` (llama-bench.cpp:394), the same function llama-server defaults to,
so leaving it alone matches the real operating point and passing a guess would not.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("aipotluck.installer.model_perf")

# The web app's generation budget -- see module docstring for the derivation. Not 160_000.
TURN_BUDGET_MS = 155_000.0

# What the web app currently requests as a ceiling per reply (GROUNDED_DECODING.max_tokens,
# overriding any model config). Kept as a documented fact rather than used in the grade: the
# thresholds above deliberately reach past it, because a model that can only just manage 1024
# tokens has no headroom for a reasoning model's thinking tokens or for that cap ever being raised.
# A model graded red can still serve today's 1024-token replies; it simply cannot do more.
APP_REQUESTED_MAX_OUTPUT_TOKENS = 1024

# Applied to the predicted turn time before comparing it to the budget. It is not padding: the
# cost model is KNOWN to run optimistic, and this is what keeps the grade's promise true anyway.
#
# Measured against real full turns on a CPU-only laptop (Qwen2.5-0.5B Q4_K_M, q8_0 K/V, via
# `llama-bench -pg`), the fit under-predicted by 1.26x at a 4,096-token input and 1.36x at 1,024.
# Two reasons, both structural rather than fixable by a better fit: llama-bench excludes
# tokenization and sampling from its numbers (its own README says so) and nothing measures the chat
# template the app renders on top; and real cost grows slightly faster than linearly past the
# deepest point the probe can afford to measure.
#
# At the old 1.25 that left a turn at N_fit really taking ~169s against a 155s budget -- the grade
# would have promised something false. 1.5 puts the same turn at ~141s. The validator
# (scripts/validate_model_perf.py) gates on exactly this: real turn at N_fit must fit the budget.
SAFETY_FACTOR_BENCH = 1.50
SAFETY_FACTOR_LIVE = 1.75

# Grade thresholds, in OUTPUT tokens: how much thinking and answer a model can actually deliver
# before the stream is cut off. Prefill is cheap next to generation, so grading on how big an input
# fits made the verdict mostly a restatement of the context size; this asks the question users care
# about instead -- how long an answer arrives intact.
OUTPUT_GREEN_MIN = 4_096   # never cut off in practice
OUTPUT_YELLOW_MIN = 1_536  # fine for shorter queries
OUTPUT_RED_MIN = 250       # only a simple, direct answer
                           # below OUTPUT_RED_MIN: cannot answer at all, refused

# Context size separately caps the grade, since a model that generates fast is still limited by how
# much it can be told. The cap is on the EFFECTIVE context -- min(model's trained context, what
# runtime sizing could actually afford on this device's memory) -- because a context this device
# cannot serve is not one the user gets.
CONTEXT_GREEN_MIN = 262_144   # above this: no cap, green is reachable
CONTEXT_YELLOW_MIN = 16_384   # 16k-256k: at most yellow
CONTEXT_RED_MIN = 4_096       # 4k-16k: at most red
                              # below CONTEXT_RED_MIN: the system prompt alone does not fit, refused

# What a single turn actually prefills. NOT the full context, deliberately: llama.cpp reuses the KV
# cache across turns of a conversation, so a chat that has grown to 100k tokens pays only for the
# new tokens each turn, not a fresh 100k prefill. Charging the full context would have cost 630s of
# prefill for a 131k model on a Jetson -- over budget before a single token came out -- and rejected
# every large-context model for having a large context, which is exactly the inversion this grading
# scheme exists to remove. Context still derates the grade, through decode depth and the cap above.
ASSUMED_PREFILL_TOKENS = 2_048

# The context depth generation is priced at. NOT each model's own full context, which is accurate
# but describes a turn nobody has: verified on a Jetson, a 3B really does fall to 3 tok/s once
# 131k tokens are actually in its KV cache, but no conversation gets there. Pricing every model at
# its own maximum also re-introduced the very inversion this scheme exists to remove -- a 131k
# model lost two thirds of its score against an otherwise-worse 32k one, so the grade tracked the
# sizing decision rather than the model.
#
# 32k is a long-but-plausible conversation (roughly an attached-file turn), it keeps every model on
# the same footing, and it stays close to measured ground: the linear depth fit was checked against
# real llama-bench runs at 16k and 32k and came back within 2%. Context has not stopped counting --
# it still slows every token up to this depth, and it still caps the grade outright.
GRADING_DECODE_DEPTH = 32_768

GRADE_GREEN = "green"
GRADE_YELLOW = "yellow"
GRADE_RED = "red"
GRADE_REFUSE = "refuse"

SOURCE_LLAMA_BENCH = "llama-bench"
SOURCE_LIVE_SERVER = "live-server"

CONFIDENCE_OK = "ok"
CONFIDENCE_LOW = "low"

# Total wall-clock the probe may spend before giving up, and the shape of each measurement. The
# prompt/gen counts are small on purpose: this is a rate measurement, and the expensive part is
# prefilling the KV cache to the requested depth, not the tokens actually timed.
PROBE_BUDGET_SECONDS = 180.0
_BENCH_N_PROMPT = 256
_BENCH_N_GEN = 32
_BENCH_REPETITIONS = 3

# Where the SHALLOW measurement is taken. Deliberately not depth 0, which is both unrepresentative
# and unsafe to anchor on. Measured on a CPU-only laptop (Qwen2.5-0.5B Q4_K_M, q8_0 K/V), decode
# cost per token was 17.3ms at depth 0 but 32.8ms at depth 512 and 33.9ms at 1024 -- it nearly
# doubles over the first few hundred tokens and then flattens, because generating at depth 0 does
# almost no attention work at all. Anchoring a straight line at that point made the fit OPTIMISTIC
# across the whole range the app actually uses (-26% at depth 512, -9% at 1024), which is the
# dangerous direction: it promises turns that then time out. Anchoring at 512 instead came back
# conservative everywhere in that range (+13% at 1024, +19% at 2048).
#
# Nothing is lost by skipping depth 0, because the web app never operates there: its system prompt
# alone is roughly 1,450 tokens before any conversation, retrieval or attachment is added.
_SHALLOW_DEPTH = 512


# Second measurement point, most-preferred first. A deeper point gives a longer lever arm for the
# depth slope; a slow device cannot afford the deepest one, and settles for less extrapolation
# accuracy in a regime where the verdict is decided by the shallow constant term anyway.
_CANDIDATE_DEPTHS = (4096, 2048, 1024)

# Relative spread across repetitions above which the measurement is called noisy. Some other
# process was probably competing for the machine.
_HIGH_VARIANCE_RATIO = 0.20

# ...but only once the spread is big enough to matter. On fast hardware a repetition lasts a few
# hundred milliseconds, where ordinary scheduler jitter clears 20% routinely -- measured on a
# Jetson Orin NX, where a GPU-served 0.5B model's whole probe ran in 7.2s and came back "low
# confidence" for a spread worth tens of milliseconds against a 155,000ms budget. Left alone that
# would have capped genuinely fast models at yellow for noise that cannot affect the verdict, so
# the relative test is paired with an absolute floor.
_MIN_MEANINGFUL_SPREAD_MS = 50.0

# Refuse to grade at all when the machine is already this busy -- the numbers would describe the
# contention, not the model.
_MAX_LOADAVG_PER_CPU = 0.5

# How long to wait for that to clear before giving up. Sized for the load a sizing probe of a large
# model leaves behind, which decays over roughly a load-average window.
_IDLE_WAIT_SECONDS = 120.0
_IDLE_POLL_SECONDS = 5.0

# Stands in for "no measurable limit from speed alone"; always clamped by ctx_size downstream.
_UNBOUNDED_TOKENS = 1 << 30


class ModelPerfError(RuntimeError):
    """Raised when a trustworthy measurement could not be produced. Callers treat this as "skip
    grading this time", never as fatal to the pull it is part of -- same posture as
    ModelSizingError, for the same reason: a missing grade is a worse user experience than a
    failed install, but only slightly."""


@dataclass
class BenchPoint:
    """One measurement, at one KV-cache depth. Costs are milliseconds per token.

    `*_effective_depth` is the average depth the timed tokens actually sat at, not the nominal
    `-d` value: a prompt-processing test at depth D processes its tokens across depths D..D+n, so
    the representative depth is D + n/2. The correction is small at these token counts but it is
    free, and getting it wrong biases the fitted slope rather than just adding noise.
    """

    depth: int
    prefill_ms_per_token: float
    decode_ms_per_token: float
    prefill_effective_depth: float
    decode_effective_depth: float
    noisy: bool


@dataclass
class PerfFit:
    """The fitted cost model. All coefficients are milliseconds; `b` and `e` are ms per token of
    KV-cache depth."""

    decode_base_ms: float  # a
    decode_depth_ms: float  # b
    prefill_base_ms: float  # c
    prefill_depth_ms: float  # e
    load_ms: float
    source: str
    confidence: str


@dataclass
class PerfResult:
    n_out: int            # output tokens that fit in the budget, at this model's full context
    grade: str            # the worse of output_grade and context_cap
    output_grade: str     # what n_out alone earns
    context_cap: str      # the best grade this effective context size allows
    decode_tokens_per_second: float  # quoted at grading_depth, not at ctx_size
    grading_depth: int    # the context depth n_out and the rate were priced at
    ctx_size: int
    fit: PerfFit
    reason: str


# Worst-first, so `min(..., key=GRADE_SEVERITY.get)` picks the more pessimistic of two grades.
GRADE_SEVERITY = {GRADE_REFUSE: 0, GRADE_RED: 1, GRADE_YELLOW: 2, GRADE_GREEN: 3}


def grade_for_output(n_out: int) -> str:
    """Grade from how many output tokens fit in the budget."""
    if n_out >= OUTPUT_GREEN_MIN:
        return GRADE_GREEN
    if n_out >= OUTPUT_YELLOW_MIN:
        return GRADE_YELLOW
    if n_out >= OUTPUT_RED_MIN:
        return GRADE_RED
    return GRADE_REFUSE


def context_grade_cap(ctx_size: int) -> str:
    """The best grade this effective context size can earn, however fast the model generates."""
    if ctx_size > CONTEXT_GREEN_MIN:
        return GRADE_GREEN
    if ctx_size >= CONTEXT_YELLOW_MIN:
        return GRADE_YELLOW
    if ctx_size >= CONTEXT_RED_MIN:
        return GRADE_RED
    return GRADE_REFUSE


def worst(*grades: str) -> str:
    return min(grades, key=lambda g: GRADE_SEVERITY[g])


def machine_is_too_busy() -> bool:
    """True when the 1-minute load average says something else is already using this machine. A
    measurement taken now would describe the contention rather than the model."""
    try:
        one_minute = os.getloadavg()[0]
    except (OSError, AttributeError):  # not available on every platform
        return False
    cpus = os.cpu_count() or 1
    return one_minute > _MAX_LOADAVG_PER_CPU * cpus


def wait_until_idle(timeout_seconds: float = _IDLE_WAIT_SECONDS) -> bool:
    """Waits for the machine to go quiet, up to `timeout_seconds`. Returns whether it did.

    Bailing the moment load looks high is wrong when the load is OURS. Benchmarking several models
    in a row re-sizes each one first -- a full model load apiece -- so a one-minute load average
    still carries the previous model's probe long after it has exited. Measured on a Jetson: the
    first model graded, then every later one was skipped as "busy" by work this command had just
    finished doing. Waiting turns a self-inflicted skip into a short pause, while still declining
    to measure a machine that is genuinely occupied by something else.
    """
    deadline = time.monotonic() + timeout_seconds
    while machine_is_too_busy():
        if time.monotonic() >= deadline:
            return False
        log.info("Waiting for this machine to go idle before measuring...")
        time.sleep(_IDLE_POLL_SECONDS)
    return True


def predict_turn_ms(fit: PerfFit, n_in: int, n_out: int, decode_depth: int | None = None) -> float:
    """Predicted wall-clock for one turn: cold load, prefilling `n_in` tokens, then generating
    `n_out` tokens. `decode_depth` is the context depth generation runs at, defaulting to `n_in`
    -- grading passes the model's full context there, since that is the slowest depth a turn will
    ever generate at."""
    depth = n_in if decode_depth is None else decode_depth
    prefill = fit.prefill_base_ms * n_in + fit.prefill_depth_ms * n_in * n_in / 2.0
    decode = n_out * decode_ms_per_token(fit, depth)
    return fit.load_ms + prefill + decode


def decode_ms_per_token(fit: PerfFit, depth: int) -> float:
    """Cost of one generated token at a given KV-cache depth."""
    return fit.decode_base_ms + fit.decode_depth_ms * depth


def grading_decode_depth(ctx_size: int) -> int:
    """The depth generation is priced at: a long-but-plausible conversation, or the whole context
    when the model cannot even hold that much."""
    return min(GRADING_DECODE_DEPTH, ctx_size)


def solve_n_out(fit: PerfFit, budget_ms: float, ctx_size: int) -> int:
    """How many output tokens fit in `budget_ms`, generating at `grading_decode_depth(ctx_size)`.

    Generation is modelled at a constant rate for the whole reply. Integrating upward from the
    starting depth would charge for depth the reply may never reach, and the point here is a
    comparable figure rather than a worst case already covered by the safety factor.

    Returns 0 when the cold load plus the turn's prefill already exhaust the budget -- there is no
    answer at all in that case, not a short one.
    """
    prefill = (
        fit.prefill_base_ms * ASSUMED_PREFILL_TOKENS
        + fit.prefill_depth_ms * ASSUMED_PREFILL_TOKENS * ASSUMED_PREFILL_TOKENS / 2.0
    )
    remaining = budget_ms - fit.load_ms - prefill
    if remaining <= 0:
        return 0
    per_token = decode_ms_per_token(fit, grading_decode_depth(ctx_size))
    if per_token <= 0:
        return _UNBOUNDED_TOKENS
    return max(0, int(remaining / per_token))


def compute_grade(fit: PerfFit, ctx_size: int) -> PerfResult:
    """Turns a fit into a verdict. Never raises -- a grading step that could itself fail would be
    worse than the problem it exists to catch.

    Two independent dimensions, and the worse one wins:

    - **How much it can say.** `n_out` is the number of output tokens that fit in the budget while
      generating at this model's full context depth. That is the question users actually feel: a
      reply cut off mid-thought is the failure, and prefill is cheap next to generation.
    - **How much it can be told.** `ctx_size` is the effective context -- what runtime sizing could
      afford on this device's memory, already capped at the model's trained context. A model that
      generates quickly is still limited by how much context it can hold, so it caps the grade.
    """
    safety = SAFETY_FACTOR_LIVE if fit.source == SOURCE_LIVE_SERVER else SAFETY_FACTOR_BENCH
    effective_budget = TURN_BUDGET_MS / safety

    depth = grading_decode_depth(ctx_size)
    n_out = solve_n_out(fit, effective_budget, ctx_size)
    output_grade = grade_for_output(n_out)
    context_cap = context_grade_cap(ctx_size)
    grade = worst(output_grade, context_cap)

    # Green is a promise that heavy use is safe. A measurement already known to be noisy, or one
    # taken without a depth point at all, cannot support a promise -- and a fit with no depth point
    # has both slopes pinned at zero, which is exactly the assumption that would manufacture one.
    capped_by_confidence = grade == GRADE_GREEN and fit.confidence == CONFIDENCE_LOW
    if capped_by_confidence:
        grade = GRADE_YELLOW

    decode_ms = decode_ms_per_token(fit, depth)
    decode_tps = 1000.0 / decode_ms if decode_ms > 0 else 0.0

    if capped_by_confidence:
        reason = "measurement was too noisy or too short to certify heavy use -- re-run the benchmark when the machine is idle"
    elif n_out == 0:
        reason = (
            f"this device cannot load the model and prefill a turn inside the "
            f"{effective_budget / 1000:.0f}s budget, so no answer arrives at all"
        )
    elif worst(output_grade, context_cap) == context_cap and context_cap != output_grade:
        reason = (
            f"generation is fine ({n_out:,} tokens in budget) but a {ctx_size:,}-token context "
            f"limits this to {context_cap}"
        )
    else:
        reason = f"about {n_out:,} output tokens fit in the budget, generating at a {depth:,}-token context"

    return PerfResult(
        n_out=n_out,
        grade=grade,
        grading_depth=depth,
        output_grade=output_grade,
        context_cap=context_cap,
        decode_tokens_per_second=decode_tps,
        ctx_size=ctx_size,
        fit=fit,
        reason=reason,
    )


def _slowest_ms(entry: dict, n_tokens: int) -> float:
    """Milliseconds per token, taken from the SLOWEST repetition rather than the mean.

    llama-bench's JSON carries the raw per-repetition `samples_ns` array (llama-bench.cpp:1797),
    so the pessimistic statistic costs nothing extra. At the default 3 repetitions the slowest of
    three sits somewhere around a p75-p85 of the underlying distribution, which is the right
    direction to be wrong in for a gate that decides whether a model gets installed.
    """
    samples = entry.get("samples_ns") or []
    if not samples or n_tokens <= 0:
        raise ModelPerfError("llama-bench returned a test with no timing samples")
    return (max(float(s) for s in samples) / 1e6) / n_tokens


def _is_noisy(entry: dict) -> bool:
    samples = [float(s) for s in (entry.get("samples_ns") or [])]
    if len(samples) < 2:
        return False
    mean = sum(samples) / len(samples)
    if mean <= 0:
        return False
    spread_ns = max(samples) - min(samples)
    if spread_ns / 1e6 <= _MIN_MEANINGFUL_SPREAD_MS:
        return False
    return spread_ns / mean > _HIGH_VARIANCE_RATIO


def _parse_bench_output(stdout: str) -> list[dict]:
    start = stdout.find("[")
    if start < 0:
        raise ModelPerfError(f"llama-bench produced no JSON:\n{stdout[-2000:]}")
    try:
        entries = json.loads(stdout[start:])
    except json.JSONDecodeError as exc:
        raise ModelPerfError(f"could not parse llama-bench JSON: {exc}") from exc
    if not isinstance(entries, list) or not entries:
        raise ModelPerfError("llama-bench returned no test results")
    return entries


def build_bench_command(
    bench_binary: Path,
    model_id: str,
    *,
    depth: int,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    gpu_layers: str | int | None = None,
) -> list[str]:
    """The benchmark must run at the same operating point llama-server will actually serve at, so
    the K/V cache types and GPU-layer count come from this model's preset rather than from
    llama-bench's defaults. Thread count is deliberately omitted -- see the module docstring."""
    cmd = [
        str(bench_binary), "-hf", model_id, "--offline",
        "-p", str(_BENCH_N_PROMPT), "-n", str(_BENCH_N_GEN), "-d", str(depth),
        "-r", str(_BENCH_REPETITIONS), "-o", "json",
    ]
    if cache_type_k:
        cmd += ["-ctk", cache_type_k]
    if cache_type_v:
        cmd += ["-ctv", cache_type_v]
    # llama-server accepts "auto"/"all" here but llama-bench wants a plain count, so anything
    # non-numeric is left off and llama-bench's own auto-detection stands in.
    if gpu_layers is not None and str(gpu_layers).strip().lstrip("-").isdigit():
        cmd += ["-ngl", str(gpu_layers)]
    return cmd


def run_bench_point(
    bench_binary: Path,
    model_id: str,
    *,
    depth: int,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    gpu_layers: str | int | None = None,
    timeout: float,
) -> tuple[BenchPoint, dict]:
    """Runs llama-bench once at `depth`. One invocation yields both a prompt-processing and a
    token-generation test, so both rates come from a single model load. Returns the point and the
    raw prompt-processing entry, whose `model_filename`/`model_size` fields save us resolving the
    Hugging Face cache layout ourselves."""
    cmd = build_bench_command(
        bench_binary, model_id, depth=depth,
        cache_type_k=cache_type_k, cache_type_v=cache_type_v, gpu_layers=gpu_layers,
    )
    log.info("Benchmarking %s at depth %d", model_id, depth)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ModelPerfError(f"llama-bench timed out after {timeout:.0f}s at depth {depth}") from exc
    if proc.returncode != 0:
        raise ModelPerfError(
            f"llama-bench exited {proc.returncode} at depth {depth}:\n{(proc.stderr or '')[-2000:]}"
        )

    entries = _parse_bench_output(proc.stdout)
    prefill = next((e for e in entries if int(e.get("n_gen", 0)) == 0), None)
    decode = next((e for e in entries if int(e.get("n_prompt", 0)) == 0), None)
    if prefill is None or decode is None:
        raise ModelPerfError(f"llama-bench did not return both a prompt and a generation test at depth {depth}")

    n_prompt = int(prefill["n_prompt"])
    n_gen = int(decode["n_gen"])
    point = BenchPoint(
        depth=depth,
        prefill_ms_per_token=_slowest_ms(prefill, n_prompt),
        decode_ms_per_token=_slowest_ms(decode, n_gen),
        prefill_effective_depth=depth + n_prompt / 2.0,
        decode_effective_depth=depth + n_gen / 2.0,
        noisy=_is_noisy(prefill) or _is_noisy(decode),
    )
    return point, prefill


def _fit_line(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float]:
    """Fits `y = base + slope*x` through two points, refusing a negative slope.

    Measurement noise can make the deeper point look *faster* than the shallow one. Taking that
    at face value would extrapolate to a model that gets cheaper the longer the context is, which
    is not merely wrong but unboundedly optimistic -- it would hand back an enormous N_fit. When
    it happens, flatten the slope and keep the slower of the two costs.
    """
    if x1 <= x0:
        return max(y0, y1), 0.0
    slope = (y1 - y0) / (x1 - x0)
    if slope <= 0:
        return max(y0, y1), 0.0
    return max(0.0, y0 - slope * x0), slope


def fit_points(shallow: BenchPoint, deep: BenchPoint | None, load_ms: float, source: str) -> PerfFit:
    """Builds the cost model from one or two measured points. With only the shallow point the
    depth slopes stay at zero, which is optimistic -- callers pair that with a refusal that is
    already decided by the depth-0 numbers alone, never with a passing grade."""
    if deep is None:
        prefill_base, prefill_depth = shallow.prefill_ms_per_token, 0.0
        decode_base, decode_depth = shallow.decode_ms_per_token, 0.0
    else:
        prefill_base, prefill_depth = _fit_line(
            shallow.prefill_effective_depth, shallow.prefill_ms_per_token,
            deep.prefill_effective_depth, deep.prefill_ms_per_token,
        )
        decode_base, decode_depth = _fit_line(
            shallow.decode_effective_depth, shallow.decode_ms_per_token,
            deep.decode_effective_depth, deep.decode_ms_per_token,
        )
    noisy = shallow.noisy or (deep.noisy if deep else False)
    return PerfFit(
        decode_base_ms=decode_base,
        decode_depth_ms=decode_depth,
        prefill_base_ms=prefill_base,
        prefill_depth_ms=prefill_depth,
        load_ms=load_ms,
        source=source,
        confidence=CONFIDENCE_LOW if (noisy or deep is None) else CONFIDENCE_OK,
    )


# Enough of the file to get a stable sequential-read rate, bounded so the sample itself cannot
# become the expensive part of the probe on slow storage.
_LOAD_SAMPLE_MAX_BYTES = 128 * 1024 * 1024
_LOAD_SAMPLE_MAX_SECONDS = 2.0
_LOAD_SAMPLE_CHUNK = 4 * 1024 * 1024


def measure_cold_load_ms(model_path: Path, model_size_bytes: int) -> float | None:
    """Estimates how long loading this model costs from COLD storage.

    This matters more than it looks. `--models-max 1` (service/runner.py) means every switch
    between models is a fresh load, so cold is the common case rather than the tail, and that load
    happens inside the same 155s budget the reply does. But at pull time the file was just written
    and is sitting in the page cache, so timing a load right now would measure RAM, not disk, and
    would understate the real cost by an order of magnitude on eMMC or SD.

    So: drop the sample range from the page cache with posix_fadvise(DONTNEED) -- stdlib, and no
    root needed -- then time a real sequential read of it and scale to the whole file. Returns
    None where that syscall does not exist (macOS, Windows), leaving the caller to fall back.
    """
    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fadvise is None or dontneed is None or model_size_bytes <= 0:
        return None
    try:
        sample_target = min(model_size_bytes, _LOAD_SAMPLE_MAX_BYTES)
        with open(model_path, "rb", buffering=0) as handle:
            fadvise(handle.fileno(), 0, sample_target, dontneed)
            started = time.monotonic()
            read_bytes = 0
            while read_bytes < sample_target:
                chunk = handle.read(min(_LOAD_SAMPLE_CHUNK, sample_target - read_bytes))
                if not chunk:
                    break
                read_bytes += len(chunk)
                if time.monotonic() - started > _LOAD_SAMPLE_MAX_SECONDS:
                    break
            elapsed = time.monotonic() - started
    except OSError as exc:
        log.debug("Could not measure cold read rate for %s: %s", model_path, exc)
        return None

    if read_bytes <= 0 or elapsed <= 0:
        return None
    bytes_per_ms = read_bytes / (elapsed * 1000.0)
    return model_size_bytes / bytes_per_ms


def _estimated_bench_ms(depth: int, fit: PerfFit) -> float:
    """Roughly what a llama-bench run at `depth` will cost, used only to decide whether the budget
    can afford it. Prefilling the cache to `depth` dominates, and llama-bench pays it once per
    repetition plus once more for its warmup run."""
    passes = _BENCH_REPETITIONS + 1
    prefill_tokens = depth + _BENCH_N_PROMPT
    return (
        fit.load_ms
        + passes * prefill_tokens * fit.prefill_base_ms
        + passes * _BENCH_N_GEN * fit.decode_base_ms
    )


def probe_performance(
    bench_binary: Path,
    model_id: str,
    *,
    ctx_size: int,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    gpu_layers: str | int | None = None,
    budget_seconds: float = PROBE_BUDGET_SECONDS,
) -> PerfResult:
    """Measures `model_id` on this device and grades it, inside `budget_seconds` of wall clock.

    Adaptive by design: the shallow measurement comes first and settles the hopeless case on its
    own, because a model that cannot emit 1024 tokens at a realistic minimum context will not be
    saved by anything a deeper measurement could show. Only if it survives that do we spend the
    rest of the budget on the second, deeper point that gives the depth slope its lever arm.
    """
    if not bench_binary.exists():
        raise ModelPerfError(f"llama-bench binary not found at {bench_binary}")

    deadline = time.monotonic() + budget_seconds
    shallow, raw = run_bench_point(
        bench_binary, model_id, depth=_SHALLOW_DEPTH,
        cache_type_k=cache_type_k, cache_type_v=cache_type_v, gpu_layers=gpu_layers,
        timeout=budget_seconds,
    )

    model_size = int(raw.get("model_size") or 0)
    model_filename = raw.get("model_filename") or ""
    load_ms = 0.0
    if model_filename and model_size > 0:
        measured = measure_cold_load_ms(Path(model_filename), model_size)
        if measured is not None:
            load_ms = measured
            log.debug("Cold load estimated at %.0fms for %s", load_ms, model_id)

    shallow_fit = fit_points(shallow, None, load_ms, SOURCE_LLAMA_BENCH)
    # A shallow-only fit has no depth slope, so it OVER-estimates how much this model can generate
    # at a deep context. If even that optimistic reading cannot reach the smallest useful answer,
    # a deeper measurement can only confirm it -- so stop rather than spend another two minutes.
    if solve_n_out(shallow_fit, TURN_BUDGET_MS / SAFETY_FACTOR_BENCH, ctx_size) < OUTPUT_RED_MIN:
        log.info("%s cannot produce even a short answer in budget -- skipping the deep probe", model_id)
        return compute_grade(shallow_fit, ctx_size)

    deep: BenchPoint | None = None
    for depth in _CANDIDATE_DEPTHS:
        if depth <= _SHALLOW_DEPTH:
            continue  # no lever arm for the slope
        remaining_ms = (deadline - time.monotonic()) * 1000.0
        if _estimated_bench_ms(depth, shallow_fit) > remaining_ms:
            continue
        try:
            deep, _ = run_bench_point(
                bench_binary, model_id, depth=depth,
                cache_type_k=cache_type_k, cache_type_v=cache_type_v, gpu_layers=gpu_layers,
                timeout=max(1.0, remaining_ms / 1000.0),
            )
        except ModelPerfError as exc:
            log.warning("Deep probe at depth %d failed, falling back to the shallow fit: %s", depth, exc)
        break

    if deep is None:
        log.warning("No budget left for a deep probe of %s -- grading on the depth-0 point alone", model_id)

    return compute_grade(fit_points(shallow, deep, load_ms, SOURCE_LLAMA_BENCH), ctx_size)
