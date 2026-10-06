"""Automatic per-model runtime-parameter sizing for llama-server's router mode
(CUR-1965's follow-up).

The service runs `llama-server` in **router mode** (`aipotluck/service/runner.py`'s
`build_llama_server_args`: no `-hf`/`--model` of its own) -- a single always-running process that
loads/unloads model instances on demand, routed by the `"model"` field on each inference request
(confirmed against vendor/llama.cpp/tools/server/server-models.cpp: router mode auto-discovers
every model already in the HF cache, keyed by the exact same `repo:quant` identity
`--cache-list`/`model_pull.list_cached_models` already use, and merges a `--models-preset` INI
file's per-model overrides on top). This module computes that INI's `ctx-size`/`parallel`/
`cache-type-k`/`cache-type-v` entries for a given model -- the router itself does the actual
detection-and-swap-on-request-model-field, so this module's whole job is making sure a model has a
*correct* preset waiting for it before the router ever picks it up, not detecting the swap itself.

`ensure_preset()` is called from two places: `aipotluck-local-client pull` (always recomputes --
a deliberate re-pull is a reasonable time to re-size) and the service's own startup (only for
cached models that don't have a preset entry yet -- backfills a deleted/corrupted presets file, or
a model that reached the cache some other way).

Why the memory budget is a fixed fraction of TOTAL installed RAM, not a live "available now"
reading: available/background memory headroom fluctuates over a device's uptime (other processes,
OS cache pressure, whatever else the user runs), and a value picked once from a single live
snapshot would bake in whatever happened to be true at that moment -- exactly the kind of "it just
works until it doesn't" gap CLAUDE.md's "Runtime parameters" section warns about. Total installed
RAM is a stable hardware fact instead. `_MEMORY_BUDGET_FRACTION` (80%, matching Ollama's own
`freeMemory*80/100` eviction threshold) is the reserved headroom for the OS, other processes, and
the compute/attention scratch buffers that scale with context but aren't captured by the KV-cache
formula alone. This is a deliberate, openly-approximate policy, not a live measurement -- monitoring
a model's *actually observed* headroom over its running lifetime and adjusting from there would be
a real improvement, but it's future work, tracked here rather than attempted in this pass.
"""

from __future__ import annotations

import logging
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from aipotluck.installer import llama_errors, model_presets
from aipotluck.installer.source_build import _total_memory_gb

log = logging.getLogger("aipotluck.installer.model_sizing")

# Small enough to load fast (this is a metadata probe, not a real serving context) -- the model's
# own hparams (n_layer, n_embd_k_gqa, ...) are fixed at load time regardless of --ctx-size.
_PROBE_CTX_SIZE = 512
_PROBE_TIMEOUT_SECONDS = 120.0
_HEALTH_POLL_INTERVAL_SECONDS = 1.0

# Quantized K/V cache tensors are block-quantized (32 elements/block); a head dimension that
# doesn't divide evenly into that block size makes llama.cpp refuse to start at all -- confirmed
# against vendor/llama.cpp/src/llama-context.cpp's own startup validation, not assumed.
_KV_QUANT_BLOCK_SIZE = 32

# Bytes per KV-cache element for each supported cache type. f16 is exact (native element size);
# q8_0/q4_0 include the per-32-element fp16 scale factor llama.cpp's block quantization adds
# (q8_0: 32 int8 + 2-byte scale = 34/32; q4_0: 16 bytes of 4-bit data + 2-byte scale = 18/32).
_KV_BYTES_PER_ELEMENT = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}

# See module docstring: reserved share of TOTAL installed RAM budgeted for a model's weights + KV
# cache + scratch buffers, matching Ollama's own 80%-of-free-memory eviction threshold.
_MEMORY_BUDGET_FRACTION = 0.80

# Below this, auto-sizing gives up on being clever and just picks the floor rather than handing
# back something too small to be useful.
_MIN_CTX_SIZE = 512

# The smallest context the web app can actually serve a turn in. Its system prompt alone is roughly
# 1,450 tokens before any conversation, retrieval or attachment, so a window under this cannot hold
# one exchange at any speed. Defined here rather than in model_screen because sizing is the first
# place that can tell -- it knows the memory budget before a single token is benchmarked -- and
# model_screen imports it from here so the two can never disagree about the same contract.
MIN_SERVABLE_CTX_TOKENS = 2_048

# Why a model's sizing can be non-viable. These mirror model_screen's reject codes deliberately:
# the user sees one vocabulary regardless of which layer caught it.
REJECT_NO_ROOM_FOR_CONTEXT = "no_room_for_context"
REJECT_CONTEXT_TOO_SMALL = "context_too_small"

_BYTES_PER_UNIT = {"MiB": 1024**2, "GiB": 1024**3}


class ModelSizingError(RuntimeError):
    """A model could not be sized.

    `model_failed_to_load` separates the two reasons this happens, because only one of them is a
    verdict about the model. llama-server exiting while loading it means this device cannot run it
    at all -- there is nothing to size, and nothing a later retry at a smaller context would fix.
    Everything else (a missing binary, a timeout, output we could not parse) is a failure of the
    probe, and a model must not be rejected for that. `failure_kind` then says whether the load
    failure was an out-of-memory, from llama_errors' shared signatures."""

    def __init__(
        self, message: str, *, model_failed_to_load: bool = False,
        failure_kind: str | None = None, exit_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.model_failed_to_load = model_failed_to_load
        self.failure_kind = failure_kind
        self.exit_code = exit_code


@dataclass
class ModelProfile:
    n_ctx_train: int
    n_layer: int
    n_embd_head_k: int
    n_embd_k_gqa: int
    n_embd_v_gqa: int
    model_size_bytes: int
    total_memory_gb: float | None  # installed physical RAM -- see module docstring


@dataclass
class SizingResult:
    """`viable` is False when this device cannot give the model a servable context. The sizing is
    still returned (callers that only want numbers keep working), but `ctx_size` is then a floor
    that was never going to be enough -- see compute_sizing."""

    ctx_size: int
    parallel: int
    cache_type_k: str | None
    cache_type_v: str | None
    tuning: dict[str, str]
    viable: bool = True
    rejection_code: str | None = None
    rejection: str | None = None


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _check_health(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _drain(proc: subprocess.Popen, lines: list[str], failures: list[BaseException]) -> None:
    """Reads the child's merged stdout+stderr until EOF, on its own thread.

    Anything raised here would otherwise vanish into a dead thread: the output simply stops
    arriving, and the probe fails much later with "could not find 'n_ctx_train'" -- a message about
    the wrong thing entirely. So a reader failure is recorded and reported alongside the parse
    error instead of being left to print a bare traceback from a thread nobody is watching."""
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            lines.append(line)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad; see the docstring
        failures.append(exc)
        log.warning("Stopped reading llama-server's output early: %s", exc)


def _parse_int(text: str, key: str) -> int:
    match = re.search(rf"\b{re.escape(key)}\s*=\s*(\d+)", text)
    if not match:
        raise ModelSizingError(f"could not find {key!r} in llama-server's startup output")
    return int(match.group(1))


def _parse_int_or_list_max(text: str, key: str) -> int:
    """Handles llama.cpp's own print_info formatting: a uniform-per-layer value prints as a plain
    number, but a value that varies by layer (heterogeneous/MoE architectures) prints as
    `[a, b, c, ...]`. Taking the max across layers over-estimates memory for the smaller layers,
    which is the safe direction to be wrong in for a sizing budget."""
    match = re.search(rf"\b{re.escape(key)}\s*=\s*(\[[^\]]*\]|\d+)", text)
    if not match:
        raise ModelSizingError(f"could not find {key!r} in llama-server's startup output")
    raw = match.group(1)
    if raw.startswith("["):
        values = [int(v.strip()) for v in raw.strip("[]").split(",") if v.strip()]
        if not values:
            raise ModelSizingError(f"{key!r} parsed as an empty list in llama-server's startup output")
        return max(values)
    return int(raw)


def _parse_model_size_bytes(text: str) -> int:
    match = re.search(r"file size\s*=\s*([\d.]+)\s*(MiB|GiB)", text)
    if not match:
        raise ModelSizingError("could not find the model's 'file size' in llama-server's startup output")
    value, unit = match.groups()
    return int(float(value) * _BYTES_PER_UNIT[unit])


def probe_model_profile(
    server_binary: Path,
    *,
    model_hf: str | None = None,
    model_path: Path | None = None,
    gpu_layers: str | int | None = None,
    host: str = "127.0.0.1",
    timeout: float = _PROBE_TIMEOUT_SECONDS,
) -> ModelProfile:
    """Spawns server_binary against the already-downloaded model at a small probe context size
    (single-model mode, NOT router mode -- a throwaway process, not the supervised router), waits
    for it to report healthy (loaded), reads its own startup log for the hparams a KV-cache sizing
    decision needs, then terminates it. Raises ModelSizingError on anything that stops this from
    producing a trustworthy profile -- callers should treat that as "skip auto-sizing this time,"
    not fatal to the pull/startup it's part of, per this project's fail-loud-but-not-catastrophic
    posture for a nice-to-have layer.
    """
    if not server_binary.exists():
        raise ModelSizingError(f"llama-server binary not found at {server_binary}")
    if not model_hf and not model_path:
        raise ModelSizingError("probe_model_profile needs model_hf or model_path")

    port = _free_port(host)
    # --parallel 1 here matches what auto-sizing itself is about to configure for real (see
    # compute_sizing) -- probing at the same slot count the real deployment will use keeps this
    # representative, though unlike the live-memory design this replaced, the memory budget below
    # no longer depends on anything measured during this probe.
    # --verbose: confirmed live against a real device (CUR-1965) that llama-server's print_info:
    # hparam lines (n_ctx_train, n_embd_k_gqa, ...) this probe depends on are gated behind a
    # verbosity threshold NOT met at default settings on at least one real build -- the vendored
    # source's LOG_DEFAULT_LLAMA constant suggested INFO-level would print by default, but that
    # didn't hold in practice. --verbose ("set verbosity to infinity") is the one setting
    # guaranteed to surface them regardless of what a given build's default threshold happens to
    # be, so the probe always passes it rather than relying on a default that's proven unreliable.
    cmd = [
        str(server_binary), "--host", host, "--port", str(port), "--ctx-size", str(_PROBE_CTX_SIZE),
        "--parallel", "1", "--verbose",
    ]
    if model_path:
        cmd += ["--model", str(model_path)]
    else:
        cmd += ["-hf", str(model_hf)]
    if gpu_layers is not None:
        cmd += ["--gpu-layers", str(gpu_layers)]

    log.info("Probing model metadata for %s", model_path or model_hf)
    # errors="replace", not the default "strict": llama.cpp truncates long string values in its
    # `llama_model_loader: - kv NN: ...` preview at a fixed width WITHOUT respecting UTF-8
    # character boundaries, so a byte-level-BPE tokenizer whose token array happens to straddle
    # that cut emits a half-written character. Confirmed on a Jetson against
    # LiquidAI/LFM2.5-8B-A1B-GGUF: the `tokenizer.ggml.tokens` line ends `c4 8a c4 8a c4 2e 2e 2e`
    # -- a two-byte `Ċ` (U+010A) chopped after its lead byte, then "...". Strict decoding raised
    # UnicodeDecodeError inside the reader thread and took the whole probe down with it.
    #
    # Nothing here needs those bytes intact: this output is scanned with ASCII regexes for hparams
    # (n_ctx_train, n_embd_k_gqa, file size) and otherwise echoed into error messages, so a U+FFFD
    # in a tokenizer preview costs nothing, while a crash costs the user the model.
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", bufsize=1,
    )
    lines: list[str] = []
    drain_failures: list[BaseException] = []
    reader = threading.Thread(target=_drain, args=(proc, lines, drain_failures), daemon=True)
    reader.start()

    try:
        deadline = time.monotonic() + timeout
        healthy = False
        while time.monotonic() < deadline:
            exit_code = proc.poll()
            if exit_code is not None:
                # The excerpt is filtered rather than tailed: llama.cpp prints its real error and
                # then dumps a backtrace, so the last 2,000 characters are the frames and the
                # allocation failure that explains everything sits just above them. Keeping the
                # blind tail classified a real Jetson OOM as "some other failure" -- the evidence
                # was in the output and was truncated away before anything could read it.
                excerpt = llama_errors.diagnostic_excerpt("".join(lines))
                kind = llama_errors.classify_failure(excerpt, exit_code)
                raise ModelSizingError(
                    f"llama-server exited (code={exit_code}) while probing model metadata:\n{excerpt}",
                    model_failed_to_load=True, failure_kind=kind, exit_code=exit_code,
                )
            if _check_health(host, port):
                healthy = True
                break
            time.sleep(_HEALTH_POLL_INTERVAL_SECONDS)

        if not healthy:
            raise ModelSizingError(f"Timed out after {timeout:.0f}s probing model metadata.")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        reader.join(timeout=5)

    output = "".join(lines)
    try:
        return ModelProfile(
            n_ctx_train=_parse_int(output, "n_ctx_train"),
            n_layer=_parse_int(output, "n_layer"),
            n_embd_head_k=_parse_int(output, "n_embd_head_k"),
            n_embd_k_gqa=_parse_int_or_list_max(output, "n_embd_k_gqa"),
            n_embd_v_gqa=_parse_int_or_list_max(output, "n_embd_v_gqa"),
            model_size_bytes=_parse_model_size_bytes(output),
            total_memory_gb=_total_memory_gb(),
        )
    except ModelSizingError as exc:
        # A truncated read and a genuinely missing hparam look identical from here -- both are just
        # a key that isn't in `output`. Say which one it was, so the next person debugs the reader
        # rather than the regex.
        if drain_failures:
            raise ModelSizingError(
                f"{exc} -- llama-server's output was cut short by a read error "
                f"({drain_failures[0]!r}), so the value may well have been printed."
            ) from exc
        raise


def _kv_cache_bytes_per_token(profile: ModelProfile, cache_type_k: str, cache_type_v: str) -> float:
    return profile.n_layer * (
        profile.n_embd_k_gqa * _KV_BYTES_PER_ELEMENT[cache_type_k]
        + profile.n_embd_v_gqa * _KV_BYTES_PER_ELEMENT[cache_type_v]
    )


def compute_sizing(profile: ModelProfile) -> SizingResult:
    """Picks ctx_size, parallel, and KV cache type from a ModelProfile. Never raises -- if
    total_memory_gb couldn't be measured (non-Linux today; _total_memory_gb() only reads
    /proc/meminfo), falls back to the model's trained context with no quantization rather than
    guessing a memory budget that doesn't exist."""
    parallel = 1
    tuning = {
        "parallel": (
            "single-user local device -- 1 slot instead of llama-server's default 4 maximizes "
            "usable context for the same memory budget (CUR-1965)"
        )
    }

    if profile.total_memory_gb is None:
        tuning["ctx_size"] = (
            "could not measure total system memory on this platform -- sized to the model's own "
            f"trained context ({profile.n_ctx_train}) with no memory-budget check"
        )
        # A trained context too small to hold one turn is a fact about the model, not the device,
        # so it is still decidable here even with no memory reading to budget against.
        too_narrow = profile.n_ctx_train < MIN_SERVABLE_CTX_TOKENS
        return SizingResult(
            ctx_size=profile.n_ctx_train, parallel=parallel, cache_type_k=None, cache_type_v=None,
            tuning=tuning, viable=not too_narrow,
            rejection_code=REJECT_CONTEXT_TOO_SMALL if too_narrow else None,
            rejection=(
                f"its trained context is {profile.n_ctx_train} tokens, under the "
                f"{MIN_SERVABLE_CTX_TOKENS} a single turn needs"
            ) if too_narrow else None,
        )

    # Budgeted against TOTAL installed RAM (a stable hardware fact), not a live "available now"
    # reading -- see module docstring. The model's own resident weight footprint comes out of the
    # same budget before anything is left for KV cache, since router mode may hold this model
    # loaded for a long time regardless of what else the box is doing at sizing time.
    total_budget_bytes = profile.total_memory_gb * (1024**3) * _MEMORY_BUDGET_FRACTION
    kv_budget_bytes = total_budget_bytes - profile.model_size_bytes

    def max_ctx_for(cache_type_k: str, cache_type_v: str) -> int:
        bytes_per_token = _kv_cache_bytes_per_token(profile, cache_type_k, cache_type_v) * parallel
        if bytes_per_token <= 0 or kv_budget_bytes <= 0:
            return 0
        return int(kv_budget_bytes // bytes_per_token)

    quantizable = profile.n_embd_head_k > 0 and profile.n_embd_head_k % _KV_QUANT_BLOCK_SIZE == 0

    if not quantizable:
        cache_type_k = cache_type_v = None
        ctx = min(max_ctx_for("f16", "f16"), profile.n_ctx_train)
        tuning["cache_type_k"] = tuning["cache_type_v"] = (
            f"n_embd_head_k={profile.n_embd_head_k} is not a multiple of {_KV_QUANT_BLOCK_SIZE}, so this "
            "architecture can't use a quantized KV cache -- kept at llama-server's f16 default"
        )
    else:
        q8_ctx = max_ctx_for("q8_0", "q8_0")
        q4_ctx = max_ctx_for("q4_0", "q4_0")
        if q8_ctx >= profile.n_ctx_train:
            cache_type_k = cache_type_v = "q8_0"
            ctx = profile.n_ctx_train
            reason = (
                f"q8_0 KV cache reaches the model's full trained context ({profile.n_ctx_train}) "
                "within the available memory budget"
            )
        elif q4_ctx >= profile.n_ctx_train:
            cache_type_k = cache_type_v = "q4_0"
            ctx = profile.n_ctx_train
            reason = (
                "memory headroom is too tight for q8_0 to reach the full trained context "
                f"({profile.n_ctx_train}); q4_0 does"
            )
        elif q4_ctx >= q8_ctx:
            cache_type_k = cache_type_v = "q4_0"
            ctx = q4_ctx
            reason = (
                f"memory headroom doesn't cover the full trained context ({profile.n_ctx_train}) even "
                f"with quantized KV cache -- q4_0 fits more of it than q8_0 ({q4_ctx} vs {q8_ctx})"
            )
        else:
            cache_type_k = cache_type_v = "q8_0"
            ctx = q8_ctx
            reason = (
                f"memory headroom doesn't cover the full trained context ({profile.n_ctx_train}) even "
                f"with quantized KV cache -- sized to the largest context that fits ({ctx})"
            )
        tuning["cache_type_k"] = tuning["cache_type_v"] = reason

    model_size_gb = profile.model_size_bytes / (1024**3)

    # The verdict is taken from the context the budget ACTUALLY supports, before the floor below
    # rewrites it. Flooring a hopeless model to 512 and returning it as though it were sized was
    # the old behaviour: the preset looked ordinary, the model stayed installed, and the failure
    # only surfaced later as a benchmark reject or a turn that could not fit its own system prompt.
    fitted_ctx = ctx
    if fitted_ctx < MIN_SERVABLE_CTX_TOKENS:
        if profile.n_ctx_train < MIN_SERVABLE_CTX_TOKENS:
            # Nothing to do with this device: the model was never trained wide enough to hold one
            # turn, so no amount of memory would change the answer.
            rejection_code = REJECT_CONTEXT_TOO_SMALL
            rejection = (
                f"its trained context is {profile.n_ctx_train} tokens, under the "
                f"{MIN_SERVABLE_CTX_TOKENS} a single turn needs"
            )
        else:
            headroom_gb = (
                profile.total_memory_gb * _MEMORY_BUDGET_FRACTION - model_size_gb
            )
            shortfall = (
                f"its {model_size_gb:.2f}GB of weights leave {headroom_gb:.2f}GB"
                if headroom_gb >= 0
                else f"its {model_size_gb:.2f}GB of weights exceed the budget by {-headroom_gb:.2f}GB"
            )
            rejection_code = REJECT_NO_ROOM_FOR_CONTEXT
            rejection = (
                f"this device has {profile.total_memory_gb:.2f}GB of memory and "
                f"{shortfall} for a KV cache -- enough for about {max(fitted_ctx, 0)} tokens of "
                f"context, where one turn needs {MIN_SERVABLE_CTX_TOKENS}"
            )
    else:
        rejection_code = rejection = None

    if ctx < _MIN_CTX_SIZE:
        tuning["ctx_size"] = (
            f"memory budget only supports ~{max(ctx, 0)} tokens of context after the model's own "
            f"{model_size_gb:.2f}GB, below the {_MIN_CTX_SIZE} floor -- using the floor anyway rather "
            f"than handing back something too small to be useful ({profile.total_memory_gb:.2f}GB total, "
            f"{int(_MEMORY_BUDGET_FRACTION * 100)}% budgeted)"
        )
        ctx = _MIN_CTX_SIZE
    else:
        tuning["ctx_size"] = (
            f"largest context fitting the memory budget after the model's own {model_size_gb:.2f}GB "
            f"({profile.total_memory_gb:.2f}GB total, {int(_MEMORY_BUDGET_FRACTION * 100)}% budgeted), "
            f"capped at the model's trained context ({profile.n_ctx_train})"
        )

    return SizingResult(
        ctx_size=ctx, parallel=parallel, cache_type_k=cache_type_k, cache_type_v=cache_type_v,
        tuning=tuning, viable=rejection_code is None, rejection_code=rejection_code,
        rejection=rejection,
    )


def ensure_preset(
    server_binary: Path,
    presets_path: Path,
    model_id: str,
    *,
    model_hf: str | None = None,
    model_path: Path | None = None,
    gpu_layers: str | int | None = None,
    force: bool = False,
) -> SizingResult | None:
    """Probes+sizes `model_id` and writes its `--models-preset` INI section, unless a section
    already exists and `force` is False (the service-startup backfill path: fill gaps, don't
    redo work every restart). `pull` always passes force=True -- a deliberate re-pull is a
    reasonable time to re-size. Returns None when skipped (already present, not forced); raises
    ModelSizingError when probing/sizing fails -- callers decide whether that's fatal to whatever
    they're doing (see this module's docstring: it generally shouldn't be)."""
    if not force and model_presets.has_preset(presets_path, model_id):
        return None

    profile = probe_model_profile(server_binary, model_hf=model_hf, model_path=model_path, gpu_layers=gpu_layers)
    sizing = compute_sizing(profile)

    args: dict[str, str | None] = {"ctx-size": str(sizing.ctx_size), "parallel": str(sizing.parallel)}
    args["cache-type-k"] = sizing.cache_type_k
    args["cache-type-v"] = sizing.cache_type_v
    model_presets.write_preset(presets_path, model_id, args)
    model_presets.write_tuning(presets_path, model_id, sizing.tuning)

    return sizing
