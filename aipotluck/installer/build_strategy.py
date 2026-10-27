"""Decide whether to install a prebuilt llama.cpp release asset or build one from source.

Two independent, confirmed-real conditions force a source build -- either is sufficient:

  1. The host wants acceleration (backend != "cpu") but no prebuilt asset was ever published for
     that exact os/arch/backend combination. Every Jetson-class board hits this: it's an arm64
     host that reports "cuda" from detect_gpu_backend (nvidia-smi is real there), but llama.cpp's
     release CI has never published a linux-arm64-cuda asset. Falling back to the linux-arm64-cpu
     asset here (like fetch.resolve_asset does for a bare "no backend match" case) would silently
     throw away the GPU -- so this case is intercepted before fetch.resolve_asset ever runs.

  2. The best available prebuilt asset for this os/arch/backend needs a newer glibc than the host
     has. llama.cpp's release CI builds its Linux arm64 assets against a
     newer base image than most arm64 boards run -- the pinned b10989 linux-arm64-cpu asset needs
     glibc 2.38, JetPack 6.2.3 / Ubuntu 22.04 ships 2.35, and the binary refuses to even start
     there. This is a distinct problem from (1): it would still bite an arm64+CPU-only host with an
     old-enough distro, no GPU involved at all.

Neither condition is Jetson-specific in mechanism, even though Jetson boards are the concrete case
that surfaced both of them at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aipotluck.installer import fetch
from aipotluck.installer.platform_detect import (
    HostProfile,
    detect_cuda_compute_capability,
    detect_glibc_version,
    glibc_satisfies,
)


@dataclass(frozen=True)
class InstallStrategy:
    use_source_build: bool
    reason: str
    asset_key: str | None = None
    asset_entry: dict[str, Any] | None = None
    # Only set when use_source_build and the host wants CUDA -- "87" (Jetson Orin), "89", etc.,
    # straight from nvidia-smi's compute_cap with the dot removed (CMAKE_CUDA_ARCHITECTURES' own
    # format). None means "wants CUDA but couldn't determine the arch" -- the caller must treat
    # that as a hard preflight failure, never guess an architecture and silently miscompile.
    cuda_arch: str | None = None


def _cuda_arch_or_none() -> str | None:
    cc = detect_cuda_compute_capability()
    return cc.replace(".", "") if cc else None


def resolve_install_strategy(profile: HostProfile, manifest: dict[str, Any]) -> InstallStrategy:
    assets = manifest["assets"]
    exact_key = profile.asset_key

    if profile.backend != "cpu" and exact_key not in assets:
        cuda_arch = _cuda_arch_or_none() if profile.backend == "cuda" else None
        return InstallStrategy(
            use_source_build=True,
            reason=f"no published release asset for {exact_key!r}",
            cuda_arch=cuda_arch,
        )

    asset_key, asset_entry = fetch.resolve_asset(manifest, profile)
    host_glibc = detect_glibc_version()
    if not glibc_satisfies(asset_entry.get("min_glibc"), host_glibc):
        host_str = ".".join(map(str, host_glibc)) if host_glibc else "unknown"
        cuda_arch = _cuda_arch_or_none() if profile.backend == "cuda" else None
        return InstallStrategy(
            use_source_build=True,
            reason=(
                f"{asset_key} needs glibc >= {asset_entry['min_glibc']}, host has {host_str}"
            ),
            cuda_arch=cuda_arch,
        )

    return InstallStrategy(
        use_source_build=False,
        reason="prebuilt asset is compatible",
        asset_key=asset_key,
        asset_entry=asset_entry,
    )
