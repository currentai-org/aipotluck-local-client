"""Our own cache of prebuilt llama-server binaries, for hosts/backends upstream doesn't publish.

build_strategy.py already tells us when no upstream release asset is viable -- every Jetson-class
arm64+CUDA board hits this, and a from-source build there takes 30-90 minutes even on a good day
(about 35 minutes at -j3 on a Jetson Orin Nano). Once we've paid that cost once for a
given (os, arch, backend, GPU architecture) combination, there's no reason to pay it again on every
matching host -- we can build the binary exactly once, archive it in the same format as the
upstream release assets, host it as a release on this project's own public repo, and let install.py
check here before ever falling back to a real compile.

Manifest shape deliberately mirrors llama_version.json's assets section exactly (release_base_url +
tag + {key: {file, sha256}}) so fetch.py's existing download_asset/extract_archive/find_binary work
completely unchanged -- this file only adds the KEY RESOLUTION (which entry matches this host),
not a second download/verify/extract implementation.

Key format includes the CUDA architecture (e.g. "linux-arm64-cuda-sm87") where upstream's own keys
never need to, because upstream doesn't publish arm64 CUDA binaries at all -- but our own cache
might eventually hold binaries for several distinct Jetson-class (or other arm64+CUDA) generations
(Xavier is SM 7.2, Orin is SM 8.7, Thor is SM 10.x), and those are not interchangeable: a binary
built for one CMAKE_CUDA_ARCHITECTURES value will not run (or run correctly) on a different one.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from aipotluck.installer.platform_detect import HostProfile, detect_glibc_version, glibc_satisfies

log = logging.getLogger("aipotluck.installer.build_cache")


def custom_asset_key(profile: HostProfile, *, cuda_arch: str | None) -> str:
    key = profile.asset_key
    if profile.backend == "cuda" and cuda_arch:
        key = f"{key}-sm{cuda_arch}"
    return key


def load_custom_manifest(path: Path) -> dict[str, Any]:
    """Empty manifest (no custom assets) if the file doesn't exist -- this file is optional, and a
    checkout without one yet must fall through to a normal source build, not error."""
    if not path.exists():
        return {"assets": {}}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def find_custom_build(
    profile: HostProfile, manifest: dict[str, Any], *, cuda_arch: str | None
) -> tuple[str, dict[str, Any]] | None:
    """(key, entry) for a cached custom build matching this exact host, or None. A CUDA host whose
    compute capability couldn't be determined never matches anything here -- there's no sensible
    key to look up, and build_strategy's own "can't safely pick an architecture" refusal is the
    right outcome for that case, the same as it would be for a real compile."""
    if profile.backend == "cuda" and cuda_arch is None:
        return None
    key = custom_asset_key(profile, cuda_arch=cuda_arch)
    assets = manifest.get("assets", {})
    entry = assets.get(key)
    if entry is None:
        return None
    host_glibc = detect_glibc_version()
    if not glibc_satisfies(entry.get("min_glibc"), host_glibc):
        log.warning(
            "Custom cached build %s exists but needs glibc >= %s (host has %s) -- falling back to "
            "a real build", key, entry.get("min_glibc"), host_glibc,
        )
        return None
    return key, entry
