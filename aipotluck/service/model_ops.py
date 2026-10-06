"""Model management, as operations rather than as CLI commands.

`pull`, `list`, `benchmark` and `remove` exist on two surfaces now: the CLI a person types, and the
HTTP API something else calls. This module is the single implementation underneath both. The split
matters more here than it usually would, because what these operations DO is subtle and hard-won --
the viability screen, the LocalScore probe, the memory freeing a benchmark needs, the preflight size
check, the OOM history a model carries. A second implementation would not stay in step with any of
it, and the way it diverged would be invisible until a remote caller got a different answer from the
same device.

So nothing here prints, and nothing here prompts. Operations return plain dicts, ready to serialize
or to render as text, and report progress through a callback the caller supplies. Decisions a human
would be asked to confirm are parameters instead: `force` on a pull that cannot fit, an explicit
DELETE for a removal. The CLI asks the question and passes the answer down; the API expects the
answer in the request.
"""

from __future__ import annotations

import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from aipotluck.installer import (
    llama_errors,
    model_localscore,
    model_perf,
    model_perf_live,
    model_perf_store,
    model_preflight,
    model_presets,
    model_screen,
    model_sizing,
)
from aipotluck.installer.model_pull import (
    DEFAULT_TIMEOUT_SECONDS as DEFAULT_PULL_TIMEOUT_SECONDS,
    ModelPullError,
    list_cached_models,
    pull_model,
)
from aipotluck.installer.source_build import _total_memory_gb
from aipotluck.service import model_health

log = logging.getLogger("aipotluck.service.model_ops")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080

Progress = Callable[[str], None]


def _noop(_message: str) -> None:
    pass


class ModelOpError(RuntimeError):
    """A failure the caller is expected to report rather than retry.

    `code` is the stable, machine-readable name (the HTTP layer maps it to a status and a client
    can branch on it); the message is for a human. `status` is the HTTP status this deserves, kept
    here so the two surfaces cannot disagree about whether something is a 404 or a 409."""

    def __init__(self, message: str, *, code: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


# --------------------------------------------------------------------------
# Shared plumbing
# --------------------------------------------------------------------------


def base_url(llama_cfg: dict) -> str:
    return f"http://{llama_cfg.get('host', DEFAULT_HOST)}:{llama_cfg.get('port', DEFAULT_PORT)}"


def require_llama(llama_cfg: dict | None) -> dict:
    if not llama_cfg or not llama_cfg.get("server_binary"):
        raise ModelOpError(
            "No llama.cpp install found -- run the installer first.",
            code="no_llama_install", status=503,
        )
    return llama_cfg


def require_presets(llama_cfg: dict) -> Path:
    presets_path = llama_cfg.get("presets_path")
    if not presets_path:
        raise ModelOpError(
            "No presets_path configured -- re-run the installer to pick this up.",
            code="no_presets_path", status=503,
        )
    return Path(presets_path)


def bench_binary(llama_cfg: dict) -> Path | None:
    """Locates llama-bench, which ships beside llama-server in every prebuilt llama.cpp release
    archive (fetch.py extracts the whole archive, it doesn't cherry-pick) and which source_build
    now builds too.

    It is looked up strictly as a SIBLING of the llama-server we actually serve with, never by
    searching the install root for any copy. A score has to describe the engine that will serve
    turns, so pairing a llama-bench from one build with a llama-server from another would quietly
    measure the wrong thing -- a worse failure than reporting it missing."""
    server_binary = llama_cfg.get("server_binary")
    if not server_binary:
        return None
    candidate = Path(server_binary).parent / ("llama-bench.exe" if os.name == "nt" else "llama-bench")
    return candidate if candidate.exists() else None


def bench_missing_reason(llama_cfg: dict) -> str:
    """Says where we looked and what that absence actually implies.

    The two cases need different advice. A source build makes llama-bench best-effort, so re-running
    the installer really can produce it. A prebuilt install always ships it, so its absence means
    `server_binary` is pointing somewhere that is not this install's own extraction -- and telling
    that user to "re-run the installer to rebuild it" sends them after a rebuild that never happens
    and cannot help."""
    server_binary = llama_cfg.get("server_binary")
    looked_in = Path(server_binary).parent if server_binary else "(no server binary recorded)"
    if llama_cfg.get("built_from_source"):
        return (
            f"llama-bench isn't in {looked_in}, so this model can't be measured. This install was "
            "built from source, where llama-bench is a best-effort extra target that can fail "
            "without failing the build -- re-run the installer to try building it again."
        )
    return (
        f"llama-bench isn't in {looked_in}, so this model can't be measured -- but every prebuilt "
        "llama.cpp archive ships it beside llama-server, so that directory is probably a stale or "
        f"partial copy rather than this install's own. Look for leftover directories under "
        f"{llama_cfg.get('install_dir', 'the install root')}, remove them, and re-run the "
        "installer to re-resolve the binary."
    )


def reload_router(llama_cfg: dict) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url(llama_cfg)}/models?reload=1", timeout=5):
            return True
    except (urllib.error.URLError, OSError):
        return False


def delete_cached_model(llama_cfg: dict, model_id: str) -> bool:
    """Removes a model's files through the running router's own `DELETE /models` (server.cpp:243 ->
    server-models.cpp's del_router_models -> common_download_remove), which stops any running
    instance and then clears the snapshot entry, its symlinks and the newly-orphaned blobs using
    llama.cpp's own cache logic rather than a reimplementation of its layout here.

    This is the only thing that actually enforces a refusal. The router auto-discovers everything
    in the HF cache and `--no-models-autoload` is a global switch rather than a per-model one, so
    merely withholding a preset would leave a refused model listed by /v1/models and selectable in
    the web app's picker -- exactly the outcome the grade exists to prevent."""
    url = f"{base_url(llama_cfg)}/models?model={urllib.parse.quote(model_id, safe='')}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="DELETE"), timeout=60) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError) as exc:
        log.warning("Could not remove %s through the running router: %s", model_id, exc)
        return False


# --------------------------------------------------------------------------
# Serialization -- one shape for both surfaces
# --------------------------------------------------------------------------


def sizing_to_dict(sizing: "model_sizing.SizingResult") -> dict[str, Any]:
    return {
        "ctx_size": sizing.ctx_size,
        "parallel": sizing.parallel,
        "cache_type_k": sizing.cache_type_k,
        "cache_type_v": sizing.cache_type_v,
        "viable": sizing.viable,
        "rejection_code": sizing.rejection_code,
        "rejection": sizing.rejection,
        "tuning": sizing.tuning,
    }


def screen_to_dict(screen: "model_screen.ScreenResult") -> dict[str, Any]:
    return {
        "rejected": screen.rejected,
        "reason_code": screen.reason_code,
        "reason": screen.reason,
        "projected_turn_ms": screen.projected_turn_ms,
        "measured": screen.measured,
        "elapsed_seconds": round(screen.elapsed_seconds, 2),
    }


def score_to_dict(score: "model_localscore.LocalScoreResult") -> dict[str, Any]:
    return {
        "localscore": round(score.score, 1),
        "band": score.band,
        "avg_prompt_tps": round(score.avg_prompt_tps, 1),
        "avg_gen_tps": round(score.avg_gen_tps, 2),
        "avg_ttft_ms": round(score.avg_ttft_ms, 1),
        "confidence": score.confidence,
        "observed_spread": round(score.observed_spread, 3),
    }


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def measure(
    llama_cfg: dict, presets_path: Path, model_id: str, *,
    budget_seconds: float, progress: Progress = _noop,
) -> tuple["model_screen.ScreenResult | None", "model_localscore.LocalScoreResult | None", str | None]:
    """Screens a model for viability, then scores it if it survives.

    Returns (screen, score, skip_reason). A None screen means no trustworthy measurement was
    possible at all -- a missing verdict, never a guessed one, with skip_reason saying why. A screen
    that rejected carries a None score, because there is no point reporting how fast a model is that
    cannot serve a turn."""
    preset_args = model_presets.read_all(presets_path).get(model_id)
    if not preset_args or not preset_args.get("ctx-size"):
        return None, None, (
            f"{model_id} has no sized preset, so there's no operating point to measure at."
        )

    # Free whatever the router holds -- not just this model. Under --models-max 1 it keeps one whole
    # model resident and llama-bench is about to load its own copy on top; on a 16GB Jetson that was
    # an outright allocation failure for models that benchmark fine on an idle box.
    model_perf_live.free_router_memory(base_url(llama_cfg))
    if not model_perf.wait_until_idle():
        return None, None, (
            "This machine is still busy, so a measurement would describe the contention rather "
            "than the model. Try again when it's idle."
        )

    ctx_size = int(preset_args["ctx-size"])
    bench = bench_binary(llama_cfg)
    if bench is None:
        return None, None, bench_missing_reason(llama_cfg)

    common = dict(
        cache_type_k=preset_args.get("cache-type-k"),
        cache_type_v=preset_args.get("cache-type-v"),
        gpu_layers=llama_cfg.get("gpu_layers"),
    )

    # The screen runs first and on its own budget: most answers are obvious, and a user about to be
    # told "no" should not wait out a full measurement to hear it.
    progress(f"screening {model_id} for viability")
    try:
        screened = model_screen.screen_model(
            bench, model_id, ctx_size=ctx_size, budget_seconds=budget_seconds, **common
        )
    except (model_perf.ModelPerfError, ValueError) as exc:
        return None, None, f"Could not screen {model_id} on this device ({exc})."
    if screened.rejected:
        return screened, None, None

    progress(f"measuring {model_id}")
    started = time.monotonic()
    try:
        cost = model_localscore.probe_cost_model(
            bench, model_id, budget_seconds=budget_seconds, **common
        )
    except (model_perf.ModelPerfError, ValueError) as exc:
        return screened, None, f"Could not score {model_id} on this device ({exc})."

    score = model_localscore.localscore(cost)
    model_perf_store.write_record(
        presets_path, model_id, score,
        llama_config=llama_cfg, preset_args=preset_args,
        engine=llama_cfg.get("tag"), ctx_size=ctx_size,
        probe_seconds=time.monotonic() - started,
    )
    return screened, score, None


# --------------------------------------------------------------------------
# The operations
# --------------------------------------------------------------------------


def _load_failure_reason(exc: "model_sizing.ModelSizingError") -> str:
    """Why a model could not be loaded, in the words the user needs."""
    if exc.failure_kind == llama_errors.FAILURE_OOM:
        return (
            "it ran out of memory while loading on this device -- the weights, and any vision "
            "projector shipped with them, do not fit"
        )
    return f"llama-server could not load it (exit code {exc.exit_code})"


def _free_router_before_loading(llama_cfg: dict) -> None:
    """Unload whatever the router is holding before we spawn something that loads a model on top.

    Under --models-max 1 the router keeps one whole model resident, so a probe starting while it
    does is competing with it for the same memory. measure() has always done this; sizing did not,
    and a 24B model on a 16GB Jetson aborted against a router holding 10.9GB of a different model.
    It matters more now than it did then: a load failure rejects the model, so getting this wrong
    would delete one that works."""
    model_perf_live.free_router_memory(base_url(llama_cfg))


def list_models(llama_cfg: dict, config_dir: Path) -> dict[str, Any]:
    """Everything this device knows about every model it has: whether it is sized, how fast it
    scored, and whether the service has held it back for running out of memory."""
    llama_cfg = require_llama(llama_cfg)
    try:
        cached = list_cached_models(Path(llama_cfg["server_binary"]))
    except ModelPullError as exc:
        raise ModelOpError(str(exc), code="cache_unreadable", status=503) from exc

    presets_path = llama_cfg.get("presets_path")
    sized = model_presets.known_model_ids(Path(presets_path)) if presets_path else set()
    presets = model_presets.read_all(Path(presets_path)) if presets_path else {}
    graded = model_perf_store.read_all(Path(presets_path)).get("models", {}) if presets_path else {}
    held_back = model_health.quarantined_models(config_dir)

    models = []
    for model_id in cached:
        record = graded.get(model_id)
        preset = presets.get(model_id, {})
        models.append({
            "id": model_id,
            "sized": model_id in sized,
            "ctx_size": int(preset["ctx-size"]) if preset.get("ctx-size", "").isdigit() else None,
            "cache_type_k": preset.get("cache-type-k"),
            "cache_type_v": preset.get("cache-type-v"),
            "score": dict(record) if record else None,
            "stale": bool(
                presets_path and record
                and model_perf_store.is_stale(Path(presets_path), record, llama_config=llama_cfg)
            ),
            "held_back": held_back.get(model_id),
        })

    return {
        "models": models,
        # The caveat travels with the numbers rather than living only in the CLI's footer: a score
        # read over HTTP is exactly as incomparable to localscore.ai as one printed in a terminal.
        "engine": {
            "tag": llama_cfg.get("tag"),
            "localscore_note": (
                "LocalScore measured on this device's own llama.cpp build, so it is not comparable "
                "to scores published at localscore.ai -- that tool ships a much older engine."
            ),
        },
    }


def check_size_before_download(model_id: str) -> dict[str, Any] | None:
    """The preflight verdict, or None when it could not be told. See model_preflight."""
    verdict = model_preflight.check_fits_in_memory(model_id, total_memory_gb=_total_memory_gb())
    if verdict is None:
        return None
    return {
        "fits": verdict.fits,
        "model_bytes": verdict.model_bytes,
        "total_memory_bytes": verdict.total_memory_bytes,
        "detail": verdict.detail,
    }


def pull(
    llama_cfg: dict, config_dir: Path, model_id: str, *,
    allow_oversized: bool = False, keep_rejected: bool = False,
    skip_benchmark: bool = False, timeout: float | None = None,
    budget_seconds: float | None = None, progress: Progress = _noop,
) -> dict[str, Any]:
    """Download a model, size it, check it can serve a turn, and score it.

    The two overrides are deliberately separate, because they answer different questions at
    different costs. `allow_oversized` says "download it even though the weights do not fit this
    machine" -- a decision made before spending anything. `keep_rejected` says "keep it even though
    it cannot serve a turn here" -- a decision made after measuring. A single flag covering both
    would mean a caller who only wanted to override the cheap pre-check silently also disabled the
    verdict that was actually measured."""
    llama_cfg = require_llama(llama_cfg)
    result: dict[str, Any] = {
        "model": model_id, "allow_oversized": allow_oversized, "keep_rejected": keep_rejected,
    }

    if not allow_oversized:
        verdict = check_size_before_download(model_id)
        result["preflight"] = verdict
        if verdict is not None and not verdict["fits"]:
            # 409: the request is well-formed and the caller can retry it verbatim with
            # allow_oversized=true.
            raise ModelOpError(verdict["detail"], code="too_large_for_device", status=409)

    progress(f"downloading {model_id}")
    try:
        pull_model(
            Path(llama_cfg["server_binary"]), model_id,
            gpu_layers=llama_cfg.get("gpu_layers"),
            timeout=timeout if timeout is not None else DEFAULT_PULL_TIMEOUT_SECONDS,
        )
    except ModelPullError as exc:
        raise ModelOpError(str(exc), code="download_failed", status=502) from exc
    result["downloaded"] = True

    presets_path = Path(llama_cfg["presets_path"]) if llama_cfg.get("presets_path") else None
    if presets_path is None:
        result["sizing"] = None
        result["warning"] = "No presets_path configured, so this model was not sized."
        result["router_reloaded"] = reload_router(llama_cfg)
        return result

    progress(f"sizing {model_id}")
    _free_router_before_loading(llama_cfg)
    try:
        sizing = model_sizing.ensure_preset(
            Path(llama_cfg["server_binary"]), presets_path, model_id,
            model_hf=model_id, gpu_layers=llama_cfg.get("gpu_layers"), force=True,
        )
    except model_sizing.ModelSizingError as exc:
        # A model that cannot be loaded at all has no sizing to fall back to and no slower mode to
        # settle for. Leaving it installed-but-unsized was how one slipped through: the router
        # discovers it anyway, picks llama-server's defaults, and the failure resurfaces as a dead
        # turn later. Only a genuine load failure rejects -- a missing binary or an unparseable
        # probe is the probe's problem, not the model's.
        if exc.model_failed_to_load and not keep_rejected:
            result["sizing"] = None
            result["sizing_error"] = str(exc)
            return _reject(
                llama_cfg, presets_path, config_dir, model_id, result,
                code=model_screen.REJECT_WONT_LOAD, reason=_load_failure_reason(exc),
            )
        sizing = None
        result["warning"] = (
            f"Automatic runtime sizing failed ({exc}) -- {model_id} will run with llama-server's "
            "own defaults until this is retried."
        )
    result["sizing"] = sizing_to_dict(sizing) if sizing else None

    # A re-pull is a fresh start: a quarantine reached against a sizing that no longer exists must
    # not outlive it, or the model stays held back for a failure that has been re-measured away.
    model_health.forget(config_dir, model_id)

    if sizing is not None and not sizing.viable and not keep_rejected:
        return _reject(
            llama_cfg, presets_path, config_dir, model_id, result,
            code=sizing.rejection_code or model_sizing.REJECT_NO_ROOM_FOR_CONTEXT,
            reason=sizing.rejection or "this device cannot give it a usable context",
        )

    if not skip_benchmark:
        progress(f"checking {model_id} can serve a turn")
        screened, score, skipped = measure(
            llama_cfg, presets_path, model_id,
            budget_seconds=budget_seconds or model_perf.PROBE_BUDGET_SECONDS,
            progress=progress,
        )
        result["screen"] = screen_to_dict(screened) if screened else None
        result["score"] = score_to_dict(score) if score else None
        if skipped:
            result["measurement_skipped"] = skipped
        if screened is not None and screened.rejected and not keep_rejected:
            return _reject(
                llama_cfg, presets_path, config_dir, model_id, result,
                code=screened.reason_code or model_screen.REJECT_TOO_SLOW, reason=screened.reason,
            )

    result["rejected"] = False
    result["router_reloaded"] = reload_router(llama_cfg)
    return result


def _reject(
    llama_cfg: dict, presets_path: Path, config_dir: Path, model_id: str,
    result: dict[str, Any], *, code: str, reason: str,
) -> dict[str, Any]:
    """Deletes a model that cannot serve a turn here, and records why.

    Deleting is what actually enforces it: the router auto-discovers the HF cache, so a model
    merely left without a preset stays listed by /v1/models and pickable in the chat UI."""
    removed = delete_cached_model(llama_cfg, model_id)
    if removed:
        model_perf_store.forget(presets_path, model_id)
        model_health.forget(config_dir, model_id)
    result.update({
        "rejected": True,
        "rejection_code": code,
        "rejection": reason,
        "removed": removed,
        "removal_note": None if removed else (
            "It is still on disk -- the service wasn't reachable to remove it. Start the service "
            "and pull it again, or keep it anyway with force."
        ),
    })
    return result


def benchmark(
    llama_cfg: dict, config_dir: Path, model_id: str | None = None, *,
    budget_seconds: float | None = None, progress: Progress = _noop,
) -> dict[str, Any]:
    """Re-measure one model, or every cached model, without re-downloading anything."""
    llama_cfg = require_llama(llama_cfg)
    presets_path = require_presets(llama_cfg)

    if model_id:
        targets = [model_id]
    else:
        try:
            targets = list_cached_models(Path(llama_cfg["server_binary"]))
        except ModelPullError as exc:
            raise ModelOpError(str(exc), code="cache_unreadable", status=503) from exc

    results = []
    for target in targets:
        entry: dict[str, Any] = {"model": target}
        # Re-size before re-measuring. The score is capped by the effective context -- the smaller
        # of what the model was trained for and what this device's memory can serve -- so a stale
        # ctx_size would silently cap it at whatever was true when the model was first pulled.
        progress(f"re-sizing {target}")
        _free_router_before_loading(llama_cfg)
        try:
            sizing = model_sizing.ensure_preset(
                Path(llama_cfg["server_binary"]), presets_path, target,
                model_hf=target, gpu_layers=llama_cfg.get("gpu_layers"), force=True,
            )
        except model_sizing.ModelSizingError as exc:
            entry["sizing"] = None
            if exc.model_failed_to_load:
                # Held back rather than deleted. The model is already installed and the user is
                # standing right here having asked for a measurement, so the gigabytes are theirs
                # to decide about -- the same split the service's own quarantine makes. `list`
                # surfaces it and `remove` acts on it.
                reason = _load_failure_reason(exc)
                model_health.quarantine(config_dir, target, reason)
                entry["held_back"] = True
                entry["rejection_code"] = model_screen.REJECT_WONT_LOAD
                entry["rejection"] = reason
                entry["sizing_error"] = str(exc)
                entry["screen"] = entry["score"] = None
                entry["measured"] = False
                results.append(entry)
                continue
            entry["warning"] = (
                f"Could not re-size {target} ({exc}) -- measured against whatever preset it had."
            )
        else:
            entry["sizing"] = sizing_to_dict(sizing) if sizing else None

        screened, score, skipped = measure(
            llama_cfg, presets_path, target,
            budget_seconds=budget_seconds or model_perf.PROBE_BUDGET_SECONDS, progress=progress,
        )
        entry["screen"] = screen_to_dict(screened) if screened else None
        entry["score"] = score_to_dict(score) if score else None
        entry["measured"] = screened is not None
        if skipped:
            entry["measurement_skipped"] = skipped
        results.append(entry)

    return {"results": results, "measured": sum(1 for r in results if r["measured"])}


def remove(llama_cfg: dict, config_dir: Path, model_id: str) -> dict[str, Any]:
    """Delete a model's weights and everything this client remembers about it."""
    llama_cfg = require_llama(llama_cfg)
    try:
        cached = list_cached_models(Path(llama_cfg["server_binary"]))
    except ModelPullError as exc:
        raise ModelOpError(str(exc), code="cache_unreadable", status=503) from exc
    if model_id not in cached:
        raise ModelOpError(
            f"{model_id} isn't in the local cache.", code="unknown_model", status=404,
        )

    record = model_health.read_record(config_dir, model_id) or {}
    if not delete_cached_model(llama_cfg, model_id):
        raise ModelOpError(
            "Could not remove it -- the service wasn't reachable. Removal goes through the "
            "running router, which also stops any instance of the model that is still loaded.",
            code="delete_failed", status=503,
        )

    presets_path = llama_cfg.get("presets_path")
    if presets_path:
        model_perf_store.forget(Path(presets_path), model_id)
    model_health.forget(config_dir, model_id)
    return {
        "model": model_id,
        "removed": True,
        "was_held_back": bool(record.get("quarantined")),
        "quarantine_reason": record.get("quarantine_reason"),
    }
