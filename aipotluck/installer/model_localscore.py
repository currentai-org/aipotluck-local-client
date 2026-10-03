"""LocalScore: a comparable number for how well this machine runs this model.

LocalScore is Mozilla Builders' open benchmark (https://www.localscore.ai). This computes it on the
llama.cpp build we already ship, using its formula and its nine scenarios verbatim, read from
localscore.cpp rather than from prose:

    score = 10 * cuberoot(avg_prompt_tps * avg_gen_tps * 1000/avg_ttft_ms)

where each average is a plain arithmetic mean across the nine scenarios, and each scenario runs
with its own context sized to prompt+generate.

## Why we project instead of running it

Running all nine honestly takes minutes -- measured on a Jetson Orin NX: 211s for a 0.5B, 1,746s
for a 14B -- because they generate 9,104 tokens between them. That is far too slow to sit inside a
`pull`. But the deepest point any scenario reaches is only 4,352 tokens, and both prefill and
decode cost are linear in KV depth over that range, so four measurements determine all nine:

    prompt_tps(P)  = 1 / (c + e*P/2)        prefill averaged over depths 0..P
    gen_tps(P,G)   = 1 / (a + b*(P + G/2))  decode averaged over depths P..P+G
    ttft_ms(P)     = (c + e*P/2) * P        the prefill wall time itself

Validated against a full nine-scenario run on the same engine and hardware, four models on a Jetson
Orin NX 16GB: errors +3.1%, +3.4%, +3.4%, -5.6%, in 6-38s against 211-1,746s (35-46x faster).

## It is not comparable to scores published on localscore.ai

The official binary is pinned to llamafile 0.9.3, which carries llama.cpp build 1500. Measured on
identical hardware, model and scenario shapes, the llama.cpp we ship is 1.7-2.5x faster -- so a
score computed here sits well above one from the official tool, and the two must not be compared.
The engine is reported with the score for exactly that reason, which is also how LocalScore's own
schema records it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass


log = logging.getLogger("aipotluck.installer.model_localscore")

# localscore.cpp's nine (prompt, generate) scenarios, with its own notes on what each represents.
SCENARIOS = [
    (1024, 16),    # 64:1  title generation
    (4096, 256),   # 16:1  content summarization
    (2048, 256),   # 8:1   lots of code to fix
    (2048, 768),   # 3:1   standard code chat
    (1024, 1024),  # 1:1   code back and forth
    (1280, 3072),  # 1:3   reasoning over code
    (384, 1152),   # 1:3   code gen with back and forth
    (64, 1024),    # 1:16  code gen/ideation
    (16, 1536),    # 1:96  QA, storytelling, reasoning
]

# The deepest KV depth any scenario reaches. Everything here is an interpolation up to this point,
# never an extrapolation beyond it, provided the probe measures at least this deep.
MAX_SCENARIO_DEPTH = max(p + g for p, g in SCENARIOS)

# LocalScore's own published interpretation bands. Reported as guidance, not used as a gate: what
# gets a model refused is model_screen's worst-case viability test, not where it lands on a curve.
BAND_EXCELLENT = 1000
BAND_GOOD = 250
BAND_POOR = 100


@dataclass
class CostModel:
    """What a probe measures: prefill cost sampled at a few prompt lengths, decode cost as a line
    in KV depth.

    Prefill is kept as measured ANCHORS rather than fitted to a line, because the real curve is not
    one. Two effects break it, and both were measured rather than assumed:

      - Fixed per-call overhead dominates tiny prompts. On a Jetson Orin NX a 16-token prompt costs
        6.93 ms/token against 1.22 at 384 -- a 5.7x difference that has nothing to do with depth.
      - On CPU there is a step at llama.cpp's default n_ubatch of 512. On an i7 laptop prefill goes
        4.33 ms/token at a 384-token prompt to 9.37 at 1024, then flattens to 10.76 by 4096.

    A straight line through two points cannot represent either. Fitting one gave -12% on the laptop
    and +3% on the Jetson -- the shape was wrong on both, the errors simply cancelled on one.
    Interpolating between measured anchors assumes no shape at all and held every reference to
    within 1.7%.
    """

    prefill_ms_per_token: dict[int, float]  # prompt length -> measured ms per token
    decode_base_ms: float                   # decode cost per token at depth 0
    decode_depth_ms: float                  # added decode cost per token of KV depth
    confidence: str = "ok"                  # "low" when the device moved under the measurement
    observed_spread: float = 0.0            # worst relative spread seen across repetitions

    @property
    def measured_depth(self) -> int:
        return max(self.prefill_ms_per_token) if self.prefill_ms_per_token else 0


@dataclass
class LocalScoreResult:
    score: float
    avg_prompt_tps: float
    avg_gen_tps: float
    avg_ttft_ms: float
    band: str
    extrapolated: bool  # True when a scenario reached past the deepest point actually measured
    confidence: str = "ok"
    observed_spread: float = 0.0


def band_for(score: float) -> str:
    if score >= BAND_EXCELLENT:
        return "excellent"
    if score >= BAND_GOOD:
        return "good"
    if score >= BAND_POOR:
        return "fair"
    return "poor"


def prefill_ms_per_token(cost: CostModel, prompt_tokens: int) -> float:
    """Piecewise-linear in prompt length between measured anchors, flat beyond the ends.

    Flat extrapolation past the last anchor is the measured behaviour, not a convenience: prefill
    cost per token plateaus once the prompt is long enough (laptop 9.37 -> 10.76 ms/token from
    1024 to 4096; Jetson 1.19 -> 1.26 over the same span).
    """
    anchors = sorted(cost.prefill_ms_per_token)
    if not anchors:
        return 0.0
    if prompt_tokens <= anchors[0]:
        return cost.prefill_ms_per_token[anchors[0]]
    if prompt_tokens >= anchors[-1]:
        return cost.prefill_ms_per_token[anchors[-1]]
    for lo, hi in zip(anchors, anchors[1:]):
        if lo <= prompt_tokens <= hi:
            span = hi - lo
            t = (prompt_tokens - lo) / span if span else 0.0
            return (cost.prefill_ms_per_token[lo]
                    + t * (cost.prefill_ms_per_token[hi] - cost.prefill_ms_per_token[lo]))
    return cost.prefill_ms_per_token[anchors[-1]]


def prefill_ms(cost: CostModel, prompt_tokens: int) -> float:
    """Wall time to prompt `prompt_tokens` into an empty cache -- which is also the TTFT the score
    uses, since the first token arrives once the prompt is in."""
    return prefill_ms_per_token(cost, prompt_tokens) * prompt_tokens


def gen_ms_per_token(cost: CostModel, prompt_tokens: int, gen_tokens: int) -> float:
    """Decode cost per token averaged over the depths this scenario's generation spans."""
    return cost.decode_base_ms + cost.decode_depth_ms * (prompt_tokens + gen_tokens / 2.0)


def localscore(cost: CostModel) -> LocalScoreResult:
    """Applies LocalScore's formula to all nine scenarios projected from a measured cost model.

    Never raises: a score is a report, and a reporting step that could fail would be worse than the
    number being approximate. A degenerate model yields 0, which reads as 'poor' and is true.
    """
    pps, gens, ttfts = [], [], []
    for prompt, gen in SCENARIOS:
        pre = prefill_ms(cost, prompt)
        if pre <= 0:
            return LocalScoreResult(0.0, 0.0, 0.0, 0.0, band_for(0.0), True, cost.confidence, cost.observed_spread)
        pps.append(prompt / pre * 1000.0)
        ttfts.append(pre)
        per_tok = gen_ms_per_token(cost, prompt, gen)
        gens.append(1000.0 / per_tok if per_tok > 0 else 0.0)

    n = len(SCENARIOS)
    avg_pp, avg_gen, avg_ttft = sum(pps)/n, sum(gens)/n, sum(ttfts)/n
    if avg_ttft <= 0 or avg_pp <= 0 or avg_gen <= 0:
        return LocalScoreResult(0.0, avg_pp, avg_gen, avg_ttft, band_for(0.0), True, cost.confidence, cost.observed_spread)

    value = 10.0 * (avg_pp * avg_gen * (1000.0 / avg_ttft)) ** (1/3)
    return LocalScoreResult(value, avg_pp, avg_gen, avg_ttft, band_for(value),
                            cost.measured_depth < max(p for p, _ in SCENARIOS),
                            cost.confidence, cost.observed_spread)


# Prompt lengths prefill is measured at. Deliberately none below 256: a 16-token prefill finishes
# in roughly 0.08s, which is too short for a DVFS-managed CPU to leave its idle clocks, so it
# measures a downclocked machine. Live, those tiny anchors swung 4.14 -> 11.36 ms/token between
# runs of the same model while the 1024 anchor held steady. They looked fine in offline analysis
# only because the reference measured them once, warm, in a single pass -- a trap worth naming.
PREFILL_ANCHORS = (256, 1024, 2048)

# Depths the decode line is fitted from. Decode genuinely IS linear in depth, so two points do.
DECODE_DEPTHS = (0, 1024)
DECODE_TOKENS = 32

# Repetitions per measurement. llama-bench amortises these over one model load, so they are far
# cheaper than repeating the whole probe, and the median of them is what gets used.
DEFAULT_REPS = 3

# Relative spread across repetitions above which the device is called thermally unstable and the
# score is reported as low confidence. A laptop drifted 32% between a cool and a warmed reference
# -- three times this method's own error -- and no projection can be more precise than the machine.
UNSTABLE_SPREAD = 0.20

CONFIDENCE_OK = "ok"
CONFIDENCE_LOW = "low"


def probe_cost_model(
    bench_binary,
    model_id: str,
    *,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    gpu_layers: str | int | None = None,
    reps: int = DEFAULT_REPS,
    timeout: float = 600.0,
) -> CostModel:
    """Measures this model on this device: prefill at several prompt lengths, decode at two depths.

    Measured COLD, with no warm-up pass, matching the official LocalScore tool's own behaviour so
    the numbers mean the same thing. The cost is that a passively-cooled device scores its peak
    rather than its sustained rate -- which is why the spread across repetitions is carried out as
    a confidence signal rather than averaged away.
    """
    from aipotluck.installer import model_perf

    def extra_args() -> list[str]:
        args: list[str] = []
        if cache_type_k:
            args += ["-ctk", cache_type_k]
        if cache_type_v:
            args += ["-ctv", cache_type_v]
        if gpu_layers is not None and str(gpu_layers).strip().lstrip("-").isdigit():
            args += ["-ngl", str(gpu_layers)]
        return args

    spreads: list[float] = []

    prompts = ",".join(str(a) for a in PREFILL_ANCHORS)
    rows = model_perf.run_bench_raw(
        bench_binary, model_id,
        ["-p", prompts, "-n", "0", "-d", "0", "-r", str(reps)] + extra_args(),
        timeout=timeout,
    )
    prefill: dict[int, float] = {}
    for r in rows:
        if r["n_gen"] == 0:
            prefill[r["n_prompt"]] = 1000.0 / model_perf.median_rate(r)
            spreads.append(model_perf.sample_spread(r))

    lo, hi = DECODE_DEPTHS
    rows = model_perf.run_bench_raw(
        bench_binary, model_id,
        ["-p", "0", "-n", str(DECODE_TOKENS), "-d", f"{lo},{hi}", "-r", str(reps)] + extra_args(),
        timeout=timeout,
    )
    decode: dict[int, float] = {}
    for r in rows:
        if r["n_gen"]:
            decode[r["n_depth"]] = 1000.0 / model_perf.median_rate(r)
            spreads.append(model_perf.sample_spread(r))

    if len(prefill) < 2 or len(decode) < 2:
        raise model_perf.ModelPerfError("llama-bench did not return the measurements the probe asked for")

    y0, y1 = decode[lo], decode[hi]
    x0, x1 = lo + DECODE_TOKENS/2, hi + DECODE_TOKENS/2
    slope = max((y1 - y0) / (x1 - x0), 0.0) if x1 != x0 else 0.0
    base = max(y0 - slope*x0, 0.0)

    worst_spread = max(spreads) if spreads else 0.0
    return CostModel(
        prefill_ms_per_token=prefill,
        decode_base_ms=base,
        decode_depth_ms=slope,
        confidence=CONFIDENCE_LOW if worst_spread > UNSTABLE_SPREAD else CONFIDENCE_OK,
        observed_spread=worst_spread,
    )
