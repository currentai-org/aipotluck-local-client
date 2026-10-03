"""Download, checksum-verify, and extract a pinned llama.cpp release asset."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from aipotluck.installer.platform_detect import HostProfile

log = logging.getLogger("aipotluck.installer.fetch")

CHUNK_SIZE = 1024 * 1024


class VersionManifestError(RuntimeError):
    pass


class ChecksumMismatchError(RuntimeError):
    pass


def load_version_manifest(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_asset(manifest: dict[str, Any], profile: HostProfile) -> tuple[str, dict[str, Any]]:
    """Return (asset_key, asset_entry), falling back to the CPU asset for
    this OS/arch if the requested backend has no manifest entry."""
    assets = manifest["assets"]
    key = profile.asset_key
    if key in assets:
        return key, assets[key]

    cpu_key = f"{profile.os_name}-{profile.arch}-cpu"
    if cpu_key in assets:
        log.warning(
            "No asset for backend=%s (key=%s); falling back to CPU asset (%s)",
            profile.backend, key, cpu_key,
        )
        return cpu_key, assets[cpu_key]

    raise VersionManifestError(
        f"No release asset available for {key!r} and no CPU fallback "
        f"({cpu_key!r}) either. Known keys: {sorted(assets)}"
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download_asset(manifest: dict[str, Any], asset_entry: dict[str, Any], dest_dir: Path) -> Path:
    """Download the asset archive into dest_dir, verifying sha256.
    Skips re-download if a valid cached copy already exists."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = asset_entry["file"]
    expected_sha = asset_entry["sha256"]
    dest_path = dest_dir / filename

    if dest_path.exists():
        actual = _sha256_file(dest_path)
        if actual == expected_sha:
            log.info("Using cached archive: %s", dest_path)
            return dest_path
        log.warning("Cached archive checksum mismatch, re-downloading: %s", dest_path)
        dest_path.unlink()

    url = f"{manifest['release_base_url']}/{manifest['tag']}/{filename}"
    log.info("Downloading %s", url)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp_path, "wb") as out:
        shutil.copyfileobj(resp, out, length=CHUNK_SIZE)

    actual = _sha256_file(tmp_path)
    if actual != expected_sha:
        tmp_path.unlink(missing_ok=True)
        raise ChecksumMismatchError(
            f"Checksum mismatch for {filename}: expected {expected_sha}, got {actual}"
        )
    tmp_path.rename(dest_path)
    log.info("Downloaded and verified: %s", dest_path)
    return dest_path


def extract_archive(archive_path: Path, extract_root: Path, tag: str, asset_key: str) -> Path:
    """Extract into extract_root/<tag>/<asset_key>/, idempotently."""
    target_dir = extract_root / tag / asset_key
    marker = target_dir / ".extracted.ok"
    if marker.exists():
        log.info("Already extracted: %s", target_dir)
        return target_dir

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == ".gz" or archive_path.name.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(target_dir, filter="data")
    elif archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(target_dir)
    else:
        raise ValueError(f"Unsupported archive type: {archive_path}")

    marker.write_text("ok\n", encoding="utf-8")
    log.info("Extracted to: %s", target_dir)
    return target_dir


# Directory-name shapes that are leftovers rather than part of an extraction -- a hand-made backup
# of a previous build parked beside the current one. A stale copy is not harmless: it holds a
# working `llama-server`, so the install keeps serving while `server_binary` quietly names an older
# engine, and anything resolved RELATIVE to that path reads as missing instead of stale. Seen live
# on a Jetson, where `llama-b10989.bak-<epoch>` won an unordered rglob and the capability check
# reported llama-bench absent while it sat in the real extraction next door.
_DEBRIS_DIR_ENDINGS = (".old", "~")
_DEBRIS_DIR_SUBSTRING = ".bak"


def _looks_like_debris(path: Path, root: Path) -> bool:
    """True when any directory between `root` and `path` is named like a leftover copy."""
    for part in path.relative_to(root).parts[:-1]:
        if part.endswith(_DEBRIS_DIR_ENDINGS) or _DEBRIS_DIR_SUBSTRING in part:
            return True
    return False


def find_binary(extract_dir: Path, binary_stem: str) -> Path:
    """Locate a binary (e.g. 'llama-server' or 'llama-bench') under extract_dir, which may itself
    contain one nested top-level folder (release archives ship as llama-<tag>/... on some
    platforms).

    The choice is deterministic -- shallowest path first, then lexicographic -- and explicitly does
    not depend on filesystem iteration order. `rglob` yields entries in whatever order the OS hands
    back, so when two copies exist the winner varies between machines, and between runs on one
    machine. That makes the resulting misconfiguration irreproducible, which is how a stale pick
    survived a re-install unnoticed. Ambiguity is also reported rather than silently resolved,
    because picking the wrong copy leaves a working install pointed at the wrong engine."""
    candidates = list(extract_dir.rglob(binary_stem)) + list(extract_dir.rglob(binary_stem + ".exe"))
    if not candidates:
        raise FileNotFoundError(f"Could not find {binary_stem} under {extract_dir}")

    # Never let the debris heuristic be the reason nothing is found: if it would reject everything,
    # fall back to the full set and let the ordering below decide.
    live = [p for p in candidates if not _looks_like_debris(p, extract_dir)] or candidates
    chosen = sorted(live, key=lambda p: (len(p.parts), p.parts))[0]

    if len(candidates) > 1:
        others = ", ".join(str(p) for p in sorted(candidates) if p != chosen)
        log.warning(
            "Found %d copies of %s under %s -- using %s. The rest (%s) are not part of this "
            "install; remove them so a later run cannot resolve to a different one.",
            len(candidates), binary_stem, extract_dir, chosen, others,
        )
    return chosen
