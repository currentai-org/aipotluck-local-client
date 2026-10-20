"""aipotluck.installer.build_strategy -- prebuilt-vs-source-build decision.

Pure logic: detect_cuda_compute_capability and detect_glibc_version are monkeypatched, never
really probed, so these pin the DECISION for a given (profile, manifest, host-glibc, host-GPU)
combination rather than re-testing the probes themselves (that's test_platform_detect.py's job).
"""

from __future__ import annotations

import json
from pathlib import Path

from aipotluck.installer import build_strategy as bs
from aipotluck.installer.platform_detect import HostProfile, detect_gpu_backend

REAL_MANIFEST = json.loads((Path(__file__).resolve().parent.parent / "llama_version.json").read_text())

MANIFEST = {
    "assets": {
        "linux-x64-cpu": {"file": "x64.tar.gz", "min_glibc": "2.34"},
        "linux-arm64-cpu": {"file": "arm64.tar.gz", "min_glibc": "2.38"},
        "linux-x64-cuda": {"file": "x64-cuda.tar.gz", "min_glibc": "2.34"},
    }
}


class TestPrebuiltIsUsedWhenCompatible:
    def test_x64_cpu_on_a_modern_host(self, monkeypatch):
        monkeypatch.setattr(bs, "detect_glibc_version", lambda: (2, 39))
        profile = HostProfile(os_name="linux", arch="x64", backend="cpu")
        strategy = bs.resolve_install_strategy(profile, MANIFEST)
        assert strategy.use_source_build is False
        assert strategy.asset_key == "linux-x64-cpu"

    def test_unknown_host_glibc_is_not_blocked(self, monkeypatch):
        monkeypatch.setattr(bs, "detect_glibc_version", lambda: None)
        profile = HostProfile(os_name="linux", arch="arm64", backend="cpu")
        strategy = bs.resolve_install_strategy(profile, MANIFEST)
        assert strategy.use_source_build is False


class TestMacsGetThePrebuiltUpstreamAsset:
    """Against the real llama_version.json, so detection and the manifest's keys must agree: a
    renamed key on one side only would send a Mac to a source build (non-cpu backend, no asset) or
    to a missing CPU fallback."""

    def test_apple_silicon_detects_metal_and_gets_upstreams_macos_arm64_build_not_a_source_build(
        self, monkeypatch
    ):
        monkeypatch.setattr(bs, "detect_glibc_version", lambda: None)  # macOS has no glibc
        profile = HostProfile(os_name="macos", arch="arm64", backend=detect_gpu_backend("macos", "arm64"))

        strategy = bs.resolve_install_strategy(profile, REAL_MANIFEST)

        assert strategy.use_source_build is False
        assert strategy.asset_key == "macos-arm64-metal"
        assert strategy.asset_entry["file"].endswith("-bin-macos-arm64.tar.gz")

    def test_intel_mac_stays_on_the_cpu_asset(self, monkeypatch):
        monkeypatch.setattr(bs, "detect_glibc_version", lambda: None)
        profile = HostProfile(os_name="macos", arch="x64", backend=detect_gpu_backend("macos", "x64"))

        strategy = bs.resolve_install_strategy(profile, REAL_MANIFEST)

        assert strategy.use_source_build is False
        assert strategy.asset_key == "macos-x64-cpu"


class TestSourceBuildForcedByMissingAsset:
    def test_jetson_class_cuda_arm64_has_no_asset_and_never_silently_downgrades_to_cpu(
        self, monkeypatch
    ):
        monkeypatch.setattr(bs, "detect_glibc_version", lambda: (2, 35))
        monkeypatch.setattr(bs, "detect_cuda_compute_capability", lambda: "8.7")
        profile = HostProfile(os_name="linux", arch="arm64", backend="cuda")
        strategy = bs.resolve_install_strategy(profile, MANIFEST)
        assert strategy.use_source_build is True
        assert strategy.cuda_arch == "87"
        assert "no published release asset" in strategy.reason
        # Never resolved to the cpu asset as a side effect -- the whole point of intercepting here.
        assert strategy.asset_key is None

    def test_missing_asset_for_non_cuda_backend_has_no_cuda_arch(self, monkeypatch):
        monkeypatch.setattr(bs, "detect_glibc_version", lambda: (2, 35))
        profile = HostProfile(os_name="linux", arch="arm64", backend="rocm")
        strategy = bs.resolve_install_strategy(profile, MANIFEST)
        assert strategy.use_source_build is True
        assert strategy.cuda_arch is None

    def test_cuda_arch_undetectable_comes_back_none_not_guessed(self, monkeypatch):
        monkeypatch.setattr(bs, "detect_glibc_version", lambda: (2, 35))
        monkeypatch.setattr(bs, "detect_cuda_compute_capability", lambda: None)
        profile = HostProfile(os_name="linux", arch="arm64", backend="cuda")
        strategy = bs.resolve_install_strategy(profile, MANIFEST)
        assert strategy.use_source_build is True
        assert strategy.cuda_arch is None


class TestSourceBuildForcedByOldGlibc:
    def test_the_confirmed_jetson_case_arm64_cpu_asset_too_old_a_host(self, monkeypatch):
        # JetPack 6.2.3: glibc 2.35 against the arm64-cpu asset's 2.38 floor. Backend here is
        # "cpu" (this test is about the glibc gate, not the missing-cuda-asset gate above).
        monkeypatch.setattr(bs, "detect_glibc_version", lambda: (2, 35))
        profile = HostProfile(os_name="linux", arch="arm64", backend="cpu")
        strategy = bs.resolve_install_strategy(profile, MANIFEST)
        assert strategy.use_source_build is True
        assert "glibc >= 2.38" in strategy.reason
        assert "host has 2.35" in strategy.reason
        assert strategy.cuda_arch is None

    def test_reason_names_unknown_host_glibc_explicitly(self, monkeypatch):
        # Can't happen in practice (an unknown host glibc never fails glibc_satisfies), but the
        # reason string must degrade sensibly if this function is ever reached anyway.
        monkeypatch.setattr(bs, "detect_glibc_version", lambda: None)
        profile = HostProfile(os_name="linux", arch="arm64", backend="cpu")
        strategy = bs.resolve_install_strategy(profile, MANIFEST)
        assert strategy.use_source_build is False  # unknown glibc never blocks -- see module docstring

    def test_old_glibc_with_cuda_backend_also_reports_cuda_arch(self, monkeypatch):
        # An x64 host old enough to miss the cuda asset's floor, but the accelerator itself is
        # real and detectable -- the source build must still target it, not fall back to CPU.
        monkeypatch.setattr(bs, "detect_glibc_version", lambda: (2, 30))
        monkeypatch.setattr(bs, "detect_cuda_compute_capability", lambda: "8.9")
        profile = HostProfile(os_name="linux", arch="x64", backend="cuda")
        strategy = bs.resolve_install_strategy(profile, MANIFEST)
        assert strategy.use_source_build is True
        assert strategy.cuda_arch == "89"
