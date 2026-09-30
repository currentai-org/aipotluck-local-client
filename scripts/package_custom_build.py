#!/usr/bin/env python3
"""Package a from-source llama-server build into this project's custom-binary-cache archive
format -- the standardized tool for "we built one of these once, now let's cache it" (see
aipotluck/installer/build_cache.py and README.md's "Building from source" section). Run this ON
the machine you just built on, right after `aipotluck-local-client`'s installer finishes a real
from-source build.

Usage:
    python3 scripts/package_custom_build.py --build-dir /path/to/llama.cpp-build

Written in Python rather than bash for the same reason the rest of this project is: it needs to
reuse real project code (build_cache.custom_asset_key for the naming convention, fetch.find_binary
for locating the binary, platform_detect for auto-deriving the key on the host that just built),
and it needs to be testable the way everything else here is (see tests/test_package_custom_build.py).

What it does, in order:
  1. Resolve the tag (from llama_version.json, unless overridden) and the cache key (auto-derived
     from THIS host's profile via the exact same code build_cache.py uses to look one up, unless
     overridden -- so packaging always produces a key the installer would actually match).
  2. Stage --build-dir's bin/ contents into a flat llama-<tag>/ directory, matching upstream's own
     release archive layout exactly (confirmed against a real llama.cpp release tarball) -- no
     nested bin/, so fetch.find_binary's recursive search works unchanged either way.
  3. Verify relocatability FOR REAL: copy the staged directory to a fresh path elsewhere and run
     `llama-server --list-devices` from there. This is not a formality -- this exact project once
     shipped a source build whose binary only worked from the absolute path it was compiled at
     (no CMAKE_INSTALL_RPATH=$ORIGIN), and that failure mode is a dynamic-linker error, not a
     clean non-zero exit that a "did it configure OK" check would ever catch. Refuses to package
     a non-relocatable build rather than silently shipping something broken to every future host
     that downloads it.
  4. tar.gz it, compute its sha256, and print the exact next steps: the `gh release` command to
     upload it, and the JSON snippet to add to llama_custom_builds.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aipotluck.installer import build_cache, fetch  # noqa: E402
from aipotluck.installer.platform_detect import (  # noqa: E402
    detect_cuda_compute_capability,
    detect_host_profile,
)

log = logging.getLogger("package_custom_build")

CUSTOM_MANIFEST_PATH = REPO_ROOT / "llama_custom_builds.json"
RELEASE_REPO = "currentai-org/aitpotluck-local-client"
_RELOCATION_FAILURE_SIGNATURES = (
    "error while loading shared libraries",  # Linux (glibc dynamic linker)
    "image not found",  # macOS (dyld)
)


class PackagingError(RuntimeError):
    pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="package_custom_build",
        description="Package a from-source llama-server build for this project's custom binary cache.",
    )
    parser.add_argument(
        "--build-dir", required=True, type=Path,
        help="The cmake build directory (contains bin/llama-server) -- what source_build.py's "
             "ensure_llama_server_built built into",
    )
    parser.add_argument(
        "--tag", default=None,
        help="llama.cpp tag this was built from (default: llama_version.json's currently pinned tag)",
    )
    parser.add_argument(
        "--key", default=None,
        help="Override the custom-cache key (default: auto-derived from THIS host's detected "
             "profile, e.g. linux-arm64-cuda-sm87 -- see build_cache.custom_asset_key)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path.cwd(),
        help="Where to write the .tar.gz (default: current directory)",
    )
    parser.add_argument(
        "--skip-relocatability-check", action="store_true",
        help="Skip verifying the binary works from a different path before packaging. Not "
             "recommended -- this is the one check that catches a missing CMAKE_INSTALL_RPATH.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def resolve_tag(args_tag: str | None) -> str:
    if args_tag:
        return args_tag
    manifest = fetch.load_version_manifest(REPO_ROOT / "llama_version.json")
    return manifest["tag"]


def resolve_key(args_key: str | None) -> str:
    if args_key:
        return args_key
    profile = detect_host_profile("auto")
    cuda_arch = None
    if profile.backend == "cuda":
        cc = detect_cuda_compute_capability()
        cuda_arch = cc.replace(".", "") if cc else None
        if cuda_arch is None:
            raise PackagingError(
                "Backend detected as cuda but this host's compute capability couldn't be "
                "determined (checked `nvidia-smi --query-gpu=compute_cap`) -- pass --key explicitly."
            )
    return build_cache.custom_asset_key(profile, cuda_arch=cuda_arch)


def stage(build_dir: Path, tag: str) -> Path:
    """Copy build_dir/bin's contents into a fresh temp llama-<tag>/ directory. Caller owns
    cleanup (the returned path's PARENT is the temp root to remove)."""
    bin_dir = build_dir / "bin"
    if not bin_dir.is_dir():
        raise PackagingError(f"No bin/ directory under {build_dir} -- did the build actually succeed?")
    staging_root = Path(tempfile.mkdtemp(prefix="package_custom_build-"))
    staged = staging_root / f"llama-{tag}"
    shutil.copytree(bin_dir, staged, symlinks=True)
    return staged


# Binaries this project actually runs, and therefore has to prove still work after being moved.
# llama-server is required; llama-bench is what model_perf grades a model with, and it is only
# absent on an older build that predates it being added to the build targets -- so it is checked
# when present rather than demanded. Both are thin launchers over their own *-impl.so, which is
# exactly the shape that breaks when an RPATH is wrong, so neither can be taken on trust.
_REQUIRED_BINARIES = ("llama-server",)
_OPTIONAL_BINARIES = ("llama-bench",)


def verify_relocatable(staged: Path) -> None:
    """Copy the staged directory to a FRESH path elsewhere and confirm the binaries we ship
    actually run from there -- proof, not inference, that they don't depend on the exact path they
    were built at."""
    targets: list[Path] = []
    for stem in _REQUIRED_BINARIES:
        try:
            targets.append(fetch.find_binary(staged, stem))
        except FileNotFoundError as exc:
            raise PackagingError(str(exc)) from exc
    for stem in _OPTIONAL_BINARIES:
        try:
            targets.append(fetch.find_binary(staged, stem))
        except FileNotFoundError:
            log.warning(
                "%s is not in this build -- packaging without it. Hosts using this archive will "
                "fall back to measuring model speed through the running server (model_perf_live).",
                stem,
            )

    with tempfile.TemporaryDirectory(prefix="relocate-check-") as tmp:
        elsewhere = Path(tmp) / "moved" / staged.name
        shutil.copytree(staged, elsewhere, symlinks=True)
        for original_binary in targets:
            moved_binary = elsewhere / original_binary.relative_to(staged)
            try:
                result = subprocess.run(
                    [str(moved_binary), "--list-devices"], capture_output=True, text=True, timeout=30,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise PackagingError(f"Could not run the relocated {original_binary.name}: {exc}") from exc

            combined = (result.stdout + result.stderr)
            if any(sig in combined.lower() for sig in _RELOCATION_FAILURE_SIGNATURES):
                raise PackagingError(
                    f"This build is not relocatable -- {original_binary.name} only works from the "
                    "exact path it was built at. This almost always means CMAKE_INSTALL_RPATH="
                    "$ORIGIN wasn't used (an old build, or a manual one predating source_build.py's "
                    f"fix). Rebuild via the installer and re-run this script.\n\nFull output from "
                    f"the relocated binary:\n{combined}"
                )
            if result.returncode != 0:
                raise PackagingError(f"The relocated {original_binary.name} exited {result.returncode}:\n{combined}")
            log.info("Relocatability check passed for %s", original_binary.name)


def package(staged: Path, tag: str, key: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"llama-{tag}-bin-{key}.tar.gz"
    with tarfile.open(out_path, "w:gz") as tf:
        tf.add(staged, arcname=staged.name)
    return out_path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_next_steps(archive_path: Path, sha256: str, tag: str, key: str) -> str:
    manifest = build_cache.load_custom_manifest(CUSTOM_MANIFEST_PATH)
    release_tag = manifest.get("tag", "custom-builds")
    release_base_url = manifest.get("release_base_url", "<release_base_url from llama_custom_builds.json>")
    snippet = json.dumps({key: {"file": archive_path.name, "sha256": sha256, "source_tag": tag}}, indent=2)
    return (
        "\n" + "=" * 70 + "\n"
        f"Packaged: {archive_path}\n"
        f"sha256:   {sha256}\n"
        + "=" * 70 + "\n\n"
        "1. Upload as a release asset (creates the release on first use, reuses it after):\n"
        f"   gh release upload {release_tag} {archive_path} --repo {RELEASE_REPO} \\\n"
        f"     || gh release create {release_tag} {archive_path} --repo {RELEASE_REPO} \\\n"
        "        --title 'Custom binary cache' \\\n"
        "        --notes 'Custom-built llama-server binaries for hosts upstream does not publish for.'\n\n"
        f"2. Add this entry to {CUSTOM_MANIFEST_PATH.relative_to(REPO_ROOT)} under \"assets\":\n"
        f"{snippet}\n\n"
        f"   Resolves to: {release_base_url}/{release_tag}/{archive_path.name}\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="[%(levelname)s] %(message)s")

    try:
        tag = resolve_tag(args.tag)
        key = resolve_key(args.key)
        log.info("Packaging %s as tag=%s key=%s", args.build_dir, tag, key)

        staged = stage(args.build_dir, tag)
        try:
            if args.skip_relocatability_check:
                log.warning("--skip-relocatability-check set -- NOT verifying the binary works from a different path")
            else:
                verify_relocatable(staged)
            archive_path = package(staged, tag, key, args.output_dir)
        finally:
            shutil.rmtree(staged.parent, ignore_errors=True)

        digest = sha256_of(archive_path)
        print(format_next_steps(archive_path, digest, tag, key))
        return 0
    except PackagingError as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
