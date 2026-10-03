"""Router housekeeping: free the memory the running llama-server is holding.

Under `--models-max 1` the router keeps one whole model resident, and a benchmark is about to load
its own copy on top. On a 16GB Jetson that was an outright allocation failure
(NvMapMemAllocInternalTagged error 12) for models that benchmark fine on an idle box -- and
unloading only the model being measured was not enough, because the router is frequently holding a
DIFFERENT one.

This module used to also measure a model by driving the router through /v1/chat/completions, as a
fallback for installs without llama-bench. That path is gone: llama-bench now ships in every
prebuilt archive and is built from source too, so measurement has one implementation rather than
two that could disagree.

`model_perf` prefers `llama-bench`, which ships in every prebuilt llama.cpp archive. Source-built
installs are the exception (`source_build.py` now builds it too, but best-effort and only on a
rebuild), and those are precisely the arm64+CUDA and old-glibc devices -- Jetson-class boards --
where a model is most likely to be too slow to use. Skipping the check there would leave it missing
exactly where it matters most, so this measures the same quantities through the one interface every
install definitely has: the router's own `/v1/chat/completions`.

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

def _get(base_url: str, path: str, *, timeout: float) -> dict | list:
    try:
        with urllib.request.urlopen(urllib.parse.urljoin(base_url, path), timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise model_perf.ModelPerfError(f"could not read {path} from {base_url}: {exc}") from exc

def free_router_memory(base_url: str, *, timeout: float = 60.0) -> list[str]:
    """Unloads every model the router currently holds, returning the ones it unloaded.

    Under `--models-max 1` the router keeps one whole model resident, and a benchmark is about to
    load its own copy on top. Unloading only the model being measured is not enough -- the router
    is frequently holding a *different* one, which then keeps its memory and makes the measurement
    fail outright. On a 16GB Jetson that was `NvMapMemAllocInternalTagged error 12`, sometimes at
    model load and sometimes at context creation, for models that benchmark fine on an idle box.

    Best-effort throughout: the router reloads whatever it needs on the next request, and a router
    that cannot be reached at all is not a reason to skip measuring.
    """
    try:
        body = _get(base_url, "/models", timeout=timeout)
    except model_perf.ModelPerfError as exc:
        log.debug("Could not list the router's models: %s", exc)
        return []

    entries = body.get("models", body.get("data", [])) if isinstance(body, dict) else body
    unloaded = []
    for entry in entries if isinstance(entries, list) else []:
        name = entry.get("name") or entry.get("id")
        status = entry.get("status")
        value = status.get("value") if isinstance(status, dict) else status
        if not name or value in (None, "unloaded", "downloaded"):
            continue
        if unload_model(base_url, name, timeout=timeout):
            unloaded.append(name)
    if unloaded:
        log.info("Freed the router's loaded model(s) before measuring: %s", ", ".join(unloaded))
    return unloaded

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
