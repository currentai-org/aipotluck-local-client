"""Persistence for `model_perf`'s grades: `llama_presets.performance.json`.

A sibling of the presets INI, exactly like `model_presets`' own `.tuning.json` -- same directory,
same read-modify-write posture, same tolerance of a corrupt or absent file. It is deliberately NOT
either of the two files already there:

- `llama_presets.ini` is parsed by llama-server itself, so an unrecognised key is llama.cpp's
  problem rather than ours. Nothing of ours belongs in it.
- `llama_presets.tuning.json` is contractually `{model_id: {param: human readable reason}}` and is
  rendered as such by `diagnostics.runtime_params` into `/status` and `/capabilities`. Structured
  numbers would break that shape for every existing reader.

## Why a device fingerprint

A grade is a claim about a model *on this machine at this llama.cpp build*, and every part of that
can change underneath it: a llama.cpp upgrade swaps the binaries, a GPU-layers change moves work
between CPU and GPU, adding RAM changes the context size the model gets sized to. A stale grade is
still useful information -- it is what the model did last time -- so it is kept and rendered as
stale rather than deleted, and re-measuring is a `benchmark` away.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import platform
from pathlib import Path
from typing import Any

from aipotluck.installer import model_perf

log = logging.getLogger("aipotluck.installer.model_perf_store")

# 2: the grade moved from "largest input that fits" to "output tokens that fit", which is a
# different number with a different meaning -- old records are dropped rather than reinterpreted.
SCHEMA_VERSION = 2


def performance_path(ini_path: Path) -> Path:
    return ini_path.with_suffix(".performance.json")


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def device_fingerprint(llama_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything that would invalidate a grade if it changed. `llama_config` is runtime.json's
    `llama_cpp` section, which carries the llama.cpp build the binaries came from and the
    GPU-layer setting every model is served with."""
    llama_config = llama_config or {}
    return {
        "llama_tag": llama_config.get("tag"),
        "install_dir": llama_config.get("install_dir"),
        "gpu_layers": llama_config.get("gpu_layers"),
        "built_from_source": bool(llama_config.get("built_from_source")),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
    }


def preset_hash(preset_args: dict[str, str | None]) -> str:
    """Identifies the operating point a grade was measured at. A model re-sized to a different
    context or K/V cache type is a different measurement, even on the same hardware."""
    canonical = json.dumps({k: v for k, v in sorted(preset_args.items()) if v is not None}, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def read_all(ini_path: Path) -> dict[str, Any]:
    path = performance_path(ini_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        # A future or unreadable schema is treated as no data rather than guessed at; the grades
        # are cheap to re-measure and a wrong grade is worse than a missing one.
        return {}
    return data


def read_record(ini_path: Path, model_id: str) -> dict[str, Any] | None:
    record = read_all(ini_path).get("models", {}).get(model_id)
    return record if isinstance(record, dict) else None


def write_record(
    ini_path: Path,
    model_id: str,
    result: model_perf.PerfResult,
    *,
    llama_config: dict[str, Any] | None = None,
    preset_args: dict[str, str | None] | None = None,
    probe_seconds: float | None = None,
) -> None:
    """Read-modify-write, same posture as model_presets.write_tuning -- another model's grade is
    never touched by a write for this one."""
    path = performance_path(ini_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = read_all(ini_path)
    data["schema_version"] = SCHEMA_VERSION
    data["device"] = device_fingerprint(llama_config)
    models = data.setdefault("models", {})
    if not isinstance(models, dict):
        models = {}
        data["models"] = models
    models[model_id] = {
        "n_out": result.n_out,
        "grade": result.grade,
        "output_grade": result.output_grade,
        "context_cap": result.context_cap,
        "decode_tokens_per_second": round(result.decode_tokens_per_second, 2),
        "ctx_size": result.ctx_size,
        "reason": result.reason,
        "source": result.fit.source,
        "confidence": result.fit.confidence,
        "fit": {
            "decode_base_ms": result.fit.decode_base_ms,
            "decode_depth_ms": result.fit.decode_depth_ms,
            "prefill_base_ms": result.fit.prefill_base_ms,
            "prefill_depth_ms": result.fit.prefill_depth_ms,
            "load_ms": result.fit.load_ms,
        },
        "preset_hash": preset_hash(preset_args or {}),
        "probe_seconds": round(probe_seconds, 1) if probe_seconds is not None else None,
        "measured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def forget(ini_path: Path, model_id: str) -> None:
    """Drops a model's grade -- used when the model itself is removed, so a later re-pull of the
    same id is measured fresh rather than inheriting a verdict."""
    path = performance_path(ini_path)
    data = read_all(ini_path)
    models = data.get("models")
    if not isinstance(models, dict) or model_id not in models:
        return
    del models[model_id]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def is_stale(
    ini_path: Path,
    record: dict[str, Any],
    *,
    llama_config: dict[str, Any] | None = None,
    preset_args: dict[str, str | None] | None = None,
) -> bool:
    """True when the device or this model's operating point has moved since the grade was taken.

    Both halves matter and they fail differently: a llama.cpp upgrade or a GPU-layers change
    invalidates every grade at once, while a re-size that picks a new context or K/V cache type
    invalidates only the model it re-sized.
    """
    if read_all(ini_path).get("device") != device_fingerprint(llama_config):
        return True
    if preset_args is not None and record.get("preset_hash") != preset_hash(preset_args):
        return True
    return False
