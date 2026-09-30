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
at depth d is `a + b*d`; integrating that over the tokens actually produced gives a closed form,
which matters here because this package is stdlib-only (`pyproject.toml`: `dependencies = []`) and
there is no numeric integrator to reach for:

    prefill token cost at depth d:  c + e*d   ->  T_prefill(n)      = c*n + e*n^2/2
    decode  token cost at depth d:  a + b*d   ->  T_decode(N, n_in) = a*N + b*(n_in*N + N^2/2)

Attention's O(n^2) term falls out of that same linear-in-depth fit, so two measurement points are
enough and no quadratic term or third point is needed. `llama-bench -d <depth>` prefills the KV
cache to a given depth before measuring, which is exactly the measurement this needs.

**The metric is `N_fit`**: the largest input, in tokens, for which a full turn still finishes
inside the budget. Substituting the fixed 1024-token output makes the constraint a quadratic in
`n`, solved in closed form by `solve_n_fit`.

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

# The web app hard-caps every reply at this many tokens, overriding any model config, so a turn's
# decode cost is bounded and knowable rather than open-ended.
MAX_OUTPUT_TOKENS = 1024

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

# Grade thresholds, in input tokens. Green is deliberately demanding: it is meant to promise that
# heavy long-context use is safe, not merely that a short chat works, so it sits above the ~32k
# turn an attached file produces. Yellow covers a RAG/web-search-grounded turn.
GREEN_MIN_TOKENS = 32_768
YELLOW_MIN_TOKENS = 8_192
RED_MIN_TOKENS = 2_048

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

# The context depth the human-readable tok/s figure is quoted at. Not the fit's depth-0 intercept,
# which no real turn ever runs at and which flatters the model -- this is a mid-range, realistic
# operating point, and quoting every model at the SAME one keeps the number comparable between them.
_REPORTED_RATE_DEPTH = 2048

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
    n_fit: int
    grade: str
    decode_tokens_per_second: float
    ctx_size: int
    fit: PerfFit
    reason: str


def grade_for(n_fit: int) -> str:
    if n_fit >= GREEN_MIN_TOKENS:
        return GRADE_GREEN
    if n_fit >= YELLOW_MIN_TOKENS:
        return GRADE_YELLOW
    if n_fit >= RED_MIN_TOKENS:
        return GRADE_RED
    return GRADE_REFUSE


def machine_is_too_busy() -> bool:
    """True when the 1-minute load average says something else is already using this machine. A
    measurement taken now would describe the contention rather than the model."""
    try:
        one_minute = os.getloadavg()[0]
    except (OSError, AttributeError):  # not available on every platform
        return False
    cpus = os.cpu_count() or 1
    return one_minute > _MAX_LOADAVG_PER_CPU * cpus


def predict_turn_ms(fit: PerfFit, n_in: int, n_out: int = MAX_OUTPUT_TOKENS) -> float:
    """Predicted wall-clock for one full turn: cold load, then prefilling `n_in` tokens, then
    generating `n_out` tokens at a depth that keeps growing as it goes."""
    prefill = fit.prefill_base_ms * n_in + fit.prefill_depth_ms * n_in * n_in / 2.0
    decode = fit.decode_base_ms * n_out + fit.decode_depth_ms * (n_in * n_out + n_out * n_out / 2.0)
    return fit.load_ms + prefill + decode


def solve_n_fit(fit: PerfFit, budget_ms: float, n_out: int = MAX_OUTPUT_TOKENS) -> int:
    """Largest `n_in` whose predicted turn still fits in `budget_ms`.

    Substituting the fixed output length into predict_turn_ms leaves a quadratic in `n_in`:

        (e/2) * n^2 + (c + b*N) * n + (load + a*N + b*N^2/2 - budget) <= 0

    Returns 0 when even an empty prompt overruns the budget -- that is the "1024 tokens alone is
    already too slow" case, and it is a refusal rather than a small number.
    """
    quad = fit.prefill_depth_ms / 2.0
    lin = fit.prefill_base_ms + fit.decode_depth_ms * n_out
    const = fit.load_ms + fit.decode_base_ms * n_out + fit.decode_depth_ms * n_out * n_out / 2.0 - budget_ms

    if const >= 0:
        return 0
    if quad <= 0:
        if lin <= 0:
            return _UNBOUNDED_TOKENS
        return int(-const / lin)
    # const < 0 and quad > 0 here, so the discriminant exceeds lin^2 and the positive root is real.
    root = (-lin + math.sqrt(lin * lin - 4.0 * quad * const)) / (2.0 * quad)
    return max(0, int(root))


def compute_grade(fit: PerfFit, ctx_size: int) -> PerfResult:
    """Turns a fit into a verdict. Never raises -- a grading step that could itself fail would be
    worse than the problem it exists to catch.

    `ctx_size` is what model_sizing.compute_sizing picked for this device's memory, and it clamps
    the answer: the app never truncates an over-long prompt, so a turn that does not fit the
    context window fails outright no matter how fast the model is. A model whose fitted context is
    below GREEN_MIN_TOKENS + MAX_OUTPUT_TOKENS therefore cannot be graded green at any speed,
    which is the honest answer for this device rather than a quirk.
    """
    safety = SAFETY_FACTOR_LIVE if fit.source == SOURCE_LIVE_SERVER else SAFETY_FACTOR_BENCH
    effective_budget = TURN_BUDGET_MS / safety

    by_speed = solve_n_fit(fit, effective_budget)
    by_context = max(0, ctx_size - MAX_OUTPUT_TOKENS)
    n_fit = max(0, min(by_speed, by_context))
    grade = grade_for(n_fit)

    # Green is a promise that heavy long-context use is safe. A measurement already known to be
    # noisy, or one taken without a depth point at all, cannot support a promise -- and a fit with
    # no depth point has both slopes pinned at zero, which is exactly the assumption that would
    # manufacture a green. Cap those at yellow rather than certifying something unmeasured.
    capped_by_confidence = grade == GRADE_GREEN and fit.confidence == CONFIDENCE_LOW
    if capped_by_confidence:
        grade = GRADE_YELLOW

    decode_ms_at_depth = fit.decode_base_ms + fit.decode_depth_ms * _REPORTED_RATE_DEPTH
    decode_tps = 1000.0 / decode_ms_at_depth if decode_ms_at_depth > 0 else 0.0

    if capped_by_confidence:
        reason = "measurement was too noisy or too short to certify heavy use -- re-run the benchmark when the machine is idle"
    elif by_speed == 0:
        reason = (
            f"a full {MAX_OUTPUT_TOKENS}-token reply alone needs about "
            f"{predict_turn_ms(fit, 0) / 1000:.0f}s, over the {effective_budget / 1000:.0f}s budget"
        )  # n_in=0 isolates the reply's own cost from any prompt
    elif by_context < by_speed:
        reason = f"limited by this device's {ctx_size}-token context for this model, not by speed"
    else:
        reason = (
            f"limited by speed at about {decode_tps:.1f} tok/s of generation "
            f"at {_REPORTED_RATE_DEPTH:,} tokens of context"
        )

    return PerfResult(
        n_fit=n_fit,
        grade=grade,
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
    if solve_n_fit(shallow_fit, TURN_BUDGET_MS / SAFETY_FACTOR_BENCH) == 0:
        # Hopeless at the shallowest context a real turn ever has. More depth can only make this
        # worse, so stop here rather than spending another two minutes confirming it.
        log.info("%s cannot finish a reply even at a minimal context -- skipping the deep probe", model_id)
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
