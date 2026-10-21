"""Platform, architecture, and GPU-backend detection.

Everything here is best-effort and non-fatal: if a probe fails or is
inconclusive, we fall back to a safe default (plain CPU asset for the
detected OS/arch). Detection never raises for a supported OS; unsupported
OSes raise UnsupportedPlatformError with a clear message.
"""

from __future__ import annotations

import platform
import shutil
import subprocess

from aipotluck.proc import NO_WINDOW
from dataclasses import dataclass


class UnsupportedPlatformError(RuntimeError):
    pass


@dataclass(frozen=True)
class HostProfile:
    os_name: str      # "linux" | "macos" | "windows"
    arch: str         # "x64" | "arm64"
    backend: str      # "cpu" | "cuda" | "vulkan" | "rocm" | "metal"

    @property
    def asset_key(self) -> str:
        return f"{self.os_name}-{self.arch}-{self.backend}"


def detect_os() -> str:
    system = platform.system()
    if system == "Linux":
        return "linux"
    if system == "Darwin":
        return "macos"
    if system == "Windows":
        return "windows"
    raise UnsupportedPlatformError(f"Unsupported OS: {system!r}")


def detect_arch() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x64"
    if machine in ("arm64", "aarch64"):
        return "arm64"
    raise UnsupportedPlatformError(f"Unsupported architecture: {machine!r}")


def _has_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def _cmd_ok(args: list[str]) -> bool:
    try:
        result = subprocess.run(
            args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, **NO_WINDOW
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def detect_gpu_backend(os_name: str, arch: str, requested: str = "auto") -> str:
    """Return one of "cpu" | "cuda" | "vulkan" | "rocm" | "metal".

    requested: "auto" to probe, or an explicit override which is returned
    as-is (validated later against the asset manifest).
    """
    if requested != "auto":
        return requested

    if os_name == "macos":
        # Nothing to probe: upstream's one Apple-silicon release asset (llama-<tag>-bin-macos-arm64)
        # is built with Metal, so an arm64 Mac gets Metal from the same download. Named "metal" so
        # the asset key, `status` and the installer's summary say what the binary actually uses,
        # rather than "cpu" (which is what this returned before, and what misled a reader into
        # thinking a Mac ran inference on the CPU). Intel Macs stay "cpu": their asset is the
        # plain CPU build as far as this project has verified.
        return "metal" if arch == "arm64" else "cpu"

    if os_name == "windows":
        # GPU asset variants for Windows are not wired into the manifest
        # yet (stubbed intentionally per project phase) -- default to CPU.
        return "cpu"

    if os_name == "linux":
        if _has_cmd("nvidia-smi") and _cmd_ok(["nvidia-smi"]):
            return "cuda"
        if _has_cmd("rocminfo") and _cmd_ok(["rocminfo"]):
            return "rocm"
        # Vulkan is a reasonable generic GPU fallback but a working Vulkan
        # loader doesn't guarantee a usable inference backend, and picking
        # it wrong is worse than CPU (which always works). Only opt in to
        # vulkan when explicitly requested by the caller.
        return "cpu"

    return "cpu"


def detect_host_profile(requested_backend: str = "auto") -> HostProfile:
    os_name = detect_os()
    arch = detect_arch()
    backend = detect_gpu_backend(os_name, arch, requested_backend)
    return HostProfile(os_name=os_name, arch=arch, backend=backend)


def detect_glibc_version() -> tuple[int, int] | None:
    """(major, minor) of the host's glibc, or None -- not glibc (musl, macOS, Windows) or the
    version string didn't parse. Never raises; callers must treat None as "unknown, don't block
    on it" rather than "definitely incompatible"."""
    lib, ver = platform.libc_ver()
    if lib != "glibc" or not ver:
        return None
    parts = ver.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None


def glibc_satisfies(min_glibc: str | None, host_glibc: tuple[int, int] | None) -> bool:
    """True unless we can PROVE the host's glibc is older than min_glibc. Either side being
    unknown (no floor recorded, or host isn't glibc-based / couldn't be read) means "don't block" --
    this check exists to catch a real, confirmed failure mode (llama.cpp's published arm64 Linux
    release asset needs glibc 2.38; JetPack 6.2.3 / Ubuntu 22.04 ships 2.35 and the binary won't
    even start), not to invent a new one out of a probe we can't complete."""
    if min_glibc is None or host_glibc is None:
        return True
    major_str, minor_str = min_glibc.split(".")
    return host_glibc >= (int(major_str), int(minor_str))


def detect_cuda_compute_capability() -> str | None:
    """The running NVIDIA GPU's compute capability as "M.m" (e.g. "8.7" on Jetson Orin, "8.9" on
    an RTX 4090), or None if nvidia-smi is absent/fails/unparseable. This is deliberately a
    separate, more detailed probe than detect_gpu_backend's plain "is cuda usable at all" check --
    it's only needed when a source build must pick a CMAKE_CUDA_ARCHITECTURES value, which a
    prebuilt-asset install never has to do."""
    if not _has_cmd("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5, text=True, **NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    value = lines[0].split(",")[0].strip()
    if "." not in value:
        return None
    return value
