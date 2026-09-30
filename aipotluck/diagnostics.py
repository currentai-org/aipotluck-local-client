"""Live system/capability fingerprint, exposed by the service's GET /capabilities endpoint
(aipotluck/service/runner.py).

Reuses exactly the same detection primitives the installer itself uses to decide prebuilt vs.
custom-cache vs. source-build (aipotluck.installer.platform_detect/build_strategy/build_cache/
source_build), so this endpoint and the installer can never silently disagree about what a host is
capable of -- this is a read-only report of the same facts install.py acts on, not a second,
parallel implementation of them.

Every section is gathered independently and defensively: one probe failing (an unreadable
/proc file, a missing manifest, a raised UnsupportedPlatformError on some future OS) must never
take down the rest of the response. This endpoint's whole purpose is to answer "why did the
installer do what it did on this box" -- including boxes that are already in a broken or unusual
state -- so it has to stay useful precisely when something else is wrong.
"""

from __future__ import annotations

import ctypes.util
import logging
import os
import platform
import shutil
from pathlib import Path
from typing import Any

from aipotluck.installer import build_cache, build_strategy, fetch, model_perf_store, model_presets, source_build
from aipotluck.installer.platform_detect import (
    HostProfile,
    UnsupportedPlatformError,
    detect_arch,
    detect_cuda_compute_capability,
    detect_glibc_version,
    detect_gpu_backend,
    detect_os,
)

log = logging.getLogger("aipotluck.diagnostics")

REPO_ROOT = Path(__file__).resolve().parent.parent
FINGERPRINT_VERSION = 1


def _safe(section_name: str, fn, *args, **kwargs) -> Any:
    """Run one detection probe; on any exception, log it and report the failure inline rather than
    letting it take down the whole /capabilities response."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see module docstring
        log.warning("diagnostics: %s probe failed: %s", section_name, exc, exc_info=True)
        return {"error": str(exc)}


def _os_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
    }
    try:
        info["name"] = detect_os()
    except UnsupportedPlatformError as exc:
        info["name"] = None
        info["detect_error"] = str(exc)

    distro_path = Path("/etc/os-release")
    if distro_path.exists():
        distro: dict[str, str] = {}
        try:
            for line in distro_path.read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    key, _, value = line.partition("=")
                    distro[key] = value.strip().strip('"')
        except OSError:
            distro = {}
        info["distro"] = distro or None
    else:
        info["distro"] = None
    return info


def _arch_info() -> dict[str, Any]:
    info: dict[str, Any] = {"raw_machine": platform.machine()}
    try:
        info["normalized"] = detect_arch()
    except UnsupportedPlatformError as exc:
        info["normalized"] = None
        info["detect_error"] = str(exc)
    return info


def _libc_info() -> dict[str, Any]:
    lib, version = platform.libc_ver()
    parsed = detect_glibc_version()
    return {
        "family": lib or None,
        "version": version or None,
        "version_tuple": list(parsed) if parsed else None,
    }


def _cpu_model() -> str | None:
    path = Path("/proc/cpuinfo")
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _cpu_info() -> dict[str, Any]:
    return {
        "logical_cores": os.cpu_count(),
        "model": _cpu_model(),
        "available_memory_gb": source_build._available_memory_gb(),
        "total_memory_gb": _total_memory_gb(),
    }


def _total_memory_gb() -> float | None:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / (1024**2)
    except (OSError, ValueError, IndexError):
        return None
    return None


def _disk_info(reference_path: Path) -> dict[str, Any]:
    return {
        "free_gb_at": str(reference_path),
        "free_gb": source_build.check_free_disk_gb(reference_path),
        "min_required_for_source_build_gb": source_build.MIN_FREE_DISK_GB,
    }


def _gpu_info(os_name: str | None, arch: str | None) -> dict[str, Any]:
    detected_backend = None
    if os_name and arch:
        detected_backend = detect_gpu_backend(os_name, arch, "auto")
    return {
        "detected_backend": detected_backend,
        "has_nvidia_smi": shutil.which("nvidia-smi") is not None,
        "cuda_compute_capability": detect_cuda_compute_capability(),
        "has_rocminfo": shutil.which("rocminfo") is not None,
    }


def _build_tools_info(*, want_cuda: bool) -> dict[str, Any]:
    nvcc = source_build.find_nvcc()
    glslc = source_build.find_glslc()
    hipcc = source_build.find_hipcc()
    return {
        "cmake": shutil.which("cmake") is not None,
        "git": shutil.which("git") is not None,
        "cxx_compiler": source_build._find_cxx_compiler(),
        "openssl_headers": source_build._has_openssl_headers(),
        "libgomp": ctypes.util.find_library("gomp") is not None,
        "nvcc": str(nvcc) if nvcc else None,
        "glslc": glslc,
        "vulkan_headers": source_build._has_vulkan_headers(),
        "hipcc": hipcc,
        "missing_apt_packages": source_build.missing_apt_packages(),
    }


def _install_strategy_info(profile: HostProfile) -> dict[str, Any]:
    manifest_path = REPO_ROOT / "llama_version.json"
    if not manifest_path.exists():
        return {"error": f"{manifest_path} not found"}
    manifest = fetch.load_version_manifest(manifest_path)
    strategy = build_strategy.resolve_install_strategy(profile, manifest)

    custom_manifest = build_cache.load_custom_manifest(REPO_ROOT / "llama_custom_builds.json")
    custom_match = None
    if strategy.use_source_build:
        custom_match = build_cache.find_custom_build(profile, custom_manifest, cuda_arch=strategy.cuda_arch)

    return {
        "pinned_tag": manifest.get("tag"),
        "resolved_backend": profile.backend,
        "use_source_build": strategy.use_source_build,
        "reason": strategy.reason,
        "upstream_asset_key": strategy.asset_key,
        "cuda_arch": strategy.cuda_arch,
        "custom_cache_key": custom_match[0] if custom_match else None,
        "custom_cache_available": custom_match is not None,
        "recommended_build": (
            {
                "jobs": source_build.recommended_jobs(want_cuda=strategy.cuda_arch is not None),
                "cmake_cuda_architectures": strategy.cuda_arch,
            }
            if strategy.use_source_build and custom_match is None
            else None
        ),
    }


_ROUTER_PARAM_FIELDS = ("host", "port", "gpu_layers", "models_dir", "models_max")

# llama.cpp preset INI keys (dashed) -> this project's own JSON field spelling (snake_case),
# matching every other field name in this API. Router-only fields (host/port/models_max/
# gpu_layers, above) never appear here -- llama-server's own preset grammar has no such keys.
_PRESET_KEY_TO_FIELD = {
    "ctx-size": "ctx_size",
    "parallel": "parallel",
    "cache-type-k": "cache_type_k",
    "cache-type-v": "cache_type_v",
}


def runtime_params(runtime_config: dict[str, Any]) -> dict[str, Any]:
    """A clean, explicit view of the llama-server router's own params plus, per model, the
    ctx_size/parallel/cache_type_k/-v aipotluck.installer.model_sizing computed for it -- and, for
    each of those, WHY, from the `{param: "reason string"}` map model_presets.read_tuning returns.
    A param's absence from a model's tuning means it's llama-server's own static default, not that
    nothing is known about it.

    Router mode (CUR-1965's follow-up) means there is no single "active model" anymore -- any
    model llama-server's router has a cache entry (or a --models-preset section) for can be
    requested at any time, so this reports EVERY sized model, keyed by id, rather than one flat set
    of fields. This is the traceability surface CLAUDE.md's "Runtime parameters" convention
    requires: any runtime parameter this project computes automatically must be visible here --
    and only here. GET /status (`aipotluck/service/runner.py`, which imports this exact function
    rather than a copy) and `aipotluck-local-client status` both surface this dict as-is, and
    /capabilities' current_install section below embeds it too -- local and remote read the same
    source, never a second one that could drift from it.
    """
    llama_cfg = runtime_config.get("llama_cpp") or {}
    params: dict[str, Any] = {field: llama_cfg.get(field) for field in _ROUTER_PARAM_FIELDS}
    params["presets_path"] = llama_cfg.get("presets_path")

    models: dict[str, Any] = {}
    presets_path_str = llama_cfg.get("presets_path")
    if presets_path_str:
        presets_path = Path(presets_path_str)
        raw_presets = model_presets.read_all(presets_path)
        tuning_by_model = model_presets.read_tuning(presets_path)
        # Speed grades live in their own sibling file rather than in the tuning map, whose contract
        # is {param: human readable reason} -- see model_perf_store's module docstring. They are
        # surfaced here for the same traceability reason the presets are: a grade that decided
        # whether a model could be installed at all should be inspectable, not printed once and lost.
        graded = model_perf_store.read_all(presets_path).get("models", {})
        for model_id, raw_args in raw_presets.items():
            model_view = {
                field: raw_args.get(key) for key, field in _PRESET_KEY_TO_FIELD.items()
            }
            model_view["tuning"] = tuning_by_model.get(model_id, {})
            record = graded.get(model_id)
            model_view["performance"] = {
                "grade": record.get("grade"),
                "n_fit": record.get("n_fit"),
                "decode_tokens_per_second": record.get("decode_tokens_per_second"),
                "source": record.get("source"),
                "confidence": record.get("confidence"),
                "measured_at": record.get("measured_at"),
                "reason": record.get("reason"),
                "stale": model_perf_store.is_stale(presets_path, record, llama_config=llama_cfg),
            } if isinstance(record, dict) else None
            models[model_id] = model_view

    params["models"] = models
    return params


def _current_install_info(runtime_config: dict[str, Any]) -> dict[str, Any]:
    llama_cfg = runtime_config.get("llama_cpp") or {}
    return {
        "tag": llama_cfg.get("tag"),
        "asset_key": llama_cfg.get("asset_key"),
        "built_from_source": llama_cfg.get("built_from_source"),
        "server_binary": llama_cfg.get("server_binary"),
        "server_binary_exists": Path(llama_cfg["server_binary"]).exists() if llama_cfg.get("server_binary") else None,
        "logged_in": bool(runtime_config.get("logged_in")),
        "runtime_params": runtime_params(runtime_config),
    }


def gather_fingerprint(*, config_dir: Path | None, runtime_config: dict[str, Any] | None) -> dict[str, Any]:
    """The full capability report. Every top-level section is independently guarded (see _safe) --
    a probe failing degrades that one section to {"error": "..."} rather than 500ing the endpoint.
    """
    runtime_config = runtime_config or {}
    os_info = _safe("os", _os_info)
    arch_info = _safe("arch", _arch_info)
    os_name = os_info.get("name") if isinstance(os_info, dict) else None
    arch = arch_info.get("normalized") if isinstance(arch_info, dict) else None

    profile = None
    if os_name and arch:
        backend = _safe("gpu_backend_for_strategy", detect_gpu_backend, os_name, arch, "auto")
        if isinstance(backend, str):
            profile = HostProfile(os_name=os_name, arch=arch, backend=backend)

    return {
        "service": "aipotluck",
        "fingerprint_version": FINGERPRINT_VERSION,
        "os": os_info,
        "arch": arch_info,
        "libc": _safe("libc", _libc_info),
        "cpu": _safe("cpu", _cpu_info),
        "disk": _safe("disk", _disk_info, config_dir or Path.home()),
        "gpu": _safe("gpu", _gpu_info, os_name, arch),
        "build_tools": _safe(
            "build_tools", _build_tools_info, want_cuda=bool(profile and profile.backend == "cuda")
        ),
        "install_strategy": (
            _safe("install_strategy", _install_strategy_info, profile)
            if profile is not None
            else {"error": "could not determine a host profile (unsupported os/arch)"}
        ),
        "current_install": _safe("current_install", _current_install_info, runtime_config),
    }
