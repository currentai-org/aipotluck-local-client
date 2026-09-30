"""Fallback speed measurement for installs with no `llama-bench`, by driving the running router.

`model_perf` prefers `llama-bench`, which ships in every prebuilt llama.cpp archive. Source-built
installs are the exception (`source_build.py` now builds it too, but best-effort and only on a
rebuild), and those are precisely the arm64+CUDA and old-glibc devices -- Jetson-class boards --
where a model is most likely to be too slow to use. Skipping the check there would leave it missing
exactly where it matters most, so this measures the same quantities through the one interface every
install definitely has: the router's own `/v1/chat/completions`.

It is a second-choice measurement and is labelled as such (`SOURCE_LIVE_SERVER`, which carries a
larger safety factor in `model_perf.compute_grade`), for three honest reasons:

- It perturbs a server the user may be talking to right now, and `--models-max 1` means it forces
  a model swap to get there.
- It measures through HTTP, JSON serialisation and the chat template, so it reads slightly slower
  than llama-bench would. That direction is safe -- it under-promises.
- Its numbers come from the server's own `timings` block rather than from a controlled harness.

## How the two points are obtained

llama-bench sets KV depth directly with `-d`. Here the prompt length *is* the depth, so two
requests with different amounts of filler give the two points:

    prompt_ms / prompt_n       -> prefill cost averaged over depths 0..n  (effective depth n/2)
    predicted_ms / predicted_n -> decode cost over depths n..n+predicted  (effective depth n + p/2)

`prompt_n` comes back from the server, so nothing here has to tokenize or even predict how long
the filler is -- the measurement is self-calibrating. `cache_prompt: false` keeps the second
request from reusing the first one's prefix and reporting a prefill that never happened.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

from aipotluck.installer import model_perf

log = logging.getLogger("aipotluck.installer.model_perf_live")

# Target prompt lengths, in words of filler (common words tokenize to roughly one token each, and
# the exact count does not matter because the server reports what it actually saw).
_SHALLOW_WORDS = 256
_DEEP_WORDS = 3072

# Short enough to keep the probe cheap, long enough to average over sampling jitter.
_GENERATED_TOKENS = 32

# Deliberately mundane and varied: a long run of one repeated token would tokenize and attend
# unrepresentatively, and a repeated prefix invites cache reuse we have explicitly disabled.
_FILLER_WORDS = (
    "the quick brown fox jumps over a lazy dog while several clever engineers review "
    "hardware notes about memory bandwidth latency throughput and cache behaviour in "
    "practical systems that serve language models to people on modest machines"
).split()


def _filler(words: int, salt: int) -> str:
    out = []
    for i in range(words):
        out.append(_FILLER_WORDS[(i + salt * 7) % len(_FILLER_WORDS)])
        if i % 12 == 11:
            out.append(str((i + salt) % 97))
    return " ".join(out)


def _post(base_url: str, path: str, payload: dict, *, timeout: float) -> dict:
    request = urllib.request.Request(
        urllib.parse.urljoin(base_url, path),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        raise model_perf.ModelPerfError(f"{path} returned HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise model_perf.ModelPerfError(f"could not reach the local router at {base_url}: {exc}") from exc


def _timed_completion(
    base_url: str, model_id: str, words: int, salt: int, *, timeout: float,
    max_tokens: int = _GENERATED_TOKENS,
) -> tuple[dict, float]:
    """`max_tokens` is short for probing -- decode rate is a rate, so a long generation would cost
    time without adding information. validate_model_perf.py overrides it to the web app's real
    1024-token cap, because there it is measuring a whole turn rather than a rate."""
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": _filler(words, salt)}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "cache_prompt": False,
        "stream": False,
    }
    started = time.monotonic()
    body = _post(base_url, "/v1/chat/completions", payload, timeout=timeout)
    wall_seconds = time.monotonic() - started

    timings = body.get("timings")
    if not isinstance(timings, dict):
        raise model_perf.ModelPerfError(
            "the router returned no `timings` block -- this llama.cpp build cannot be measured this way"
        )
    return timings, wall_seconds


def _point_from_timings(timings: dict) -> model_perf.BenchPoint:
    prompt_n = float(timings.get("prompt_n") or 0)
    prompt_ms = float(timings.get("prompt_ms") or 0)
    predicted_n = float(timings.get("predicted_n") or 0)
    predicted_ms = float(timings.get("predicted_ms") or 0)
    if prompt_n <= 0 or predicted_n <= 0 or prompt_ms <= 0 or predicted_ms <= 0:
        raise model_perf.ModelPerfError(f"the router reported unusable timings: {timings}")

    return model_perf.BenchPoint(
        depth=int(prompt_n),
        prefill_ms_per_token=prompt_ms / prompt_n,
        decode_ms_per_token=predicted_ms / predicted_n,
        prefill_effective_depth=prompt_n / 2.0,
        decode_effective_depth=prompt_n + predicted_n / 2.0,
        noisy=False,
    )


def unload_model(base_url: str, model_id: str, *, timeout: float = 30.0) -> bool:
    """Best-effort: forces the next request to pay a real cold load so it can be measured. The
    router already swaps models on demand under `--models-max 1`, so this is the same disruption
    the next differently-modelled request would have caused anyway."""
    try:
        _post(base_url, "/models/unload", {"model": model_id}, timeout=timeout)
        return True
    except model_perf.ModelPerfError as exc:
        log.debug("Could not unload %s before measuring its load time: %s", model_id, exc)
        return False


def probe_performance_live(
    base_url: str,
    model_id: str,
    *,
    ctx_size: int,
    budget_seconds: float = model_perf.PROBE_BUDGET_SECONDS,
) -> model_perf.PerfResult:
    """Measures and grades `model_id` by driving the already-running router at `base_url`."""
    deadline = time.monotonic() + budget_seconds

    # Cold load is inside the web app's turn budget and is the common case under --models-max 1,
    # so it is measured rather than assumed: unload, then attribute whatever the first request's
    # wall clock spent outside its own reported prefill and decode to loading the model.
    unload_model(base_url, model_id)
    shallow_timings, wall_seconds = _timed_completion(
        base_url, model_id, _SHALLOW_WORDS, salt=0, timeout=max(1.0, deadline - time.monotonic()),
    )
    shallow = _point_from_timings(shallow_timings)
    accounted_ms = float(shallow_timings.get("prompt_ms", 0)) + float(shallow_timings.get("predicted_ms", 0))
    load_ms = max(0.0, wall_seconds * 1000.0 - accounted_ms)

    shallow_fit = model_perf.fit_points(shallow, None, load_ms, model_perf.SOURCE_LIVE_SERVER)
    budget = model_perf.TURN_BUDGET_MS / model_perf.SAFETY_FACTOR_LIVE
    if model_perf.solve_n_out(shallow_fit, budget, ctx_size) < model_perf.OUTPUT_RED_MIN:
        log.info("%s cannot produce even a short answer in budget -- skipping the deep probe", model_id)
        return model_perf.compute_grade(shallow_fit, ctx_size)

    deep = None
    remaining = deadline - time.monotonic()
    if remaining > 0:
        try:
            deep_timings, _ = _timed_completion(
                base_url, model_id, _DEEP_WORDS, salt=1, timeout=remaining,
            )
            deep = _point_from_timings(deep_timings)
        except model_perf.ModelPerfError as exc:
            log.warning("Deep probe failed, grading on the shallow point alone: %s", exc)

    return model_perf.compute_grade(
        model_perf.fit_points(shallow, deep, load_ms, model_perf.SOURCE_LIVE_SERVER), ctx_size
    )
