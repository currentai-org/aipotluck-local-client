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

# The ceiling the web app requests per reply (GROUNDED_DECODING.max_tokens, merged over any model
# config so it always wins). Verified against a real llama.cpp server: it honours the cap exactly
# (predicted_n == max_tokens, finish_reason "length"), and hitting it is handled gracefully --
# the app appends a "say continue for the rest" notice rather than ending mid-thought.
#
# It is a REQUEST, not a contract: the app imposes no cap of its own on what it will accept, so a
# local server that ignored max_tokens would be bounded only by the 155s watchdog. For grading
# purposes the compliant case is the one that matters, and it makes this the natural green line --
# a model that can deliver this many tokens in budget is never cut off by time.

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


# Second measurement point, most-preferred first. A deeper point gives a longer lever arm for the
# depth slope; a slow device cannot afford the deepest one, and settles for less extrapolation
# accuracy in a regime where the verdict is decided by the shallow constant term anyway.

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


def decode_ms_per_token(fit: PerfFit, depth: int) -> float:
    """Cost of one generated token at a given KV-cache depth."""
    return fit.decode_base_ms + fit.decode_depth_ms * depth


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
    n_prompt: int = _BENCH_N_PROMPT,
    n_gen: int = _BENCH_N_GEN,
) -> list[str]:
    """The benchmark must run at the same operating point llama-server will actually serve at, so
    the K/V cache types and GPU-layer count come from this model's preset rather than from
    llama-bench's defaults. Thread count is deliberately omitted -- see the module docstring."""
    cmd = [
        str(bench_binary), "-hf", model_id, "--offline",
        "-p", str(n_prompt), "-n", str(n_gen), "-d", str(depth),
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


def run_bench_raw(bench_binary: Path, model_id: str, args: list[str], *, timeout: float) -> list[dict]:
    """Runs llama-bench with explicit arguments and returns its parsed JSON rows. The generic form
    behind run_bench_point, for callers that need several prompt lengths or depths from one model
    load rather than a single point."""
    cmd = [str(bench_binary), "-hf", model_id, "--offline", "-o", "json"] + args
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ModelPerfError(f"llama-bench timed out after {timeout:.0f}s") from exc
    if proc.returncode != 0:
        raise ModelPerfError(f"llama-bench exited {proc.returncode}:\n{(proc.stderr or '')[-2000:]}")
    return _parse_bench_output(proc.stdout)


def median_rate(entry: dict) -> float:
    """Tokens per second from the MEDIAN repetition, not llama-bench's mean.

    On a thermally unstable laptop three probes of one model scored 62/51/56 against a reference of
    64 -- that spread is the machine, and a mean carries its outlier straight into the fit."""
    samples = [float(x) for x in (entry.get("samples_ns") or [])]
    if not samples:
        return float(entry["avg_ts"])
    samples.sort()
    mid = len(samples) // 2
    median_ns = samples[mid] if len(samples) % 2 else (samples[mid-1] + samples[mid]) / 2
    tokens = entry["n_prompt"] or entry["n_gen"]
    return tokens / (median_ns / 1e9) if median_ns else float(entry["avg_ts"])


def sample_spread(entry: dict) -> tuple[float, float]:
    """Spread across repetitions, as (fraction of the median, absolute milliseconds).

    BOTH are needed, and returning only the fraction is a mistake this project has now made twice.
    A device moving under the measurement is the signal worth having -- a laptop measured 32%
    slower warmed than cool. But on fast hardware an individual test is tiny: a 256-token prefill
    on a Jetson GPU runs in ~77ms, where ordinary scheduler jitter clears 65% relative while being
    perhaps 50ms absolute, which cannot matter to anything. Judge on the relative figure alone and
    every fast device reports itself unstable.
    """
    samples = [float(x) for x in (entry.get("samples_ns") or [])]
    if len(samples) < 2:
        return 0.0, 0.0
    samples.sort()
    mid = len(samples) // 2
    median_ns = samples[mid] if len(samples) % 2 else (samples[mid-1] + samples[mid]) / 2
    spread_ns = samples[-1] - samples[0]
    return (spread_ns / median_ns if median_ns else 0.0), spread_ns / 1e6


def run_bench_point(
    bench_binary: Path,
    model_id: str,
    *,
    depth: int,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    gpu_layers: str | int | None = None,
    timeout: float,
    n_prompt: int = _BENCH_N_PROMPT,
    n_gen: int = _BENCH_N_GEN,
) -> tuple[BenchPoint, dict]:
    """Runs llama-bench once at `depth`. One invocation yields both a prompt-processing and a
    token-generation test, so both rates come from a single model load. Returns the point and the
    raw prompt-processing entry, whose `model_filename`/`model_size` fields save us resolving the
    Hugging Face cache layout ourselves."""
    cmd = build_bench_command(
        bench_binary, model_id, depth=depth,
        cache_type_k=cache_type_k, cache_type_v=cache_type_v, gpu_layers=gpu_layers,
        n_prompt=n_prompt, n_gen=n_gen,
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


