"""aipotluck.installer.platform_detect -- OS/arch/GPU-backend detection.

Every test monkeypatches the underlying `platform`/subprocess probes rather than trusting whatever
happens to be true of the machine running the suite, so these pass identically in CI and on any
dev laptop.
"""

from __future__ import annotations

import pytest

from aipotluck.installer import platform_detect as pd


class TestDetectOs:
    @pytest.mark.parametrize(
        ("system_value", "expected"),
        [("Linux", "linux"), ("Darwin", "macos"), ("Windows", "windows")],
    )
    def test_maps_known_platforms(self, monkeypatch, system_value, expected):
        monkeypatch.setattr(pd.platform, "system", lambda: system_value)
        assert pd.detect_os() == expected

    def test_unknown_platform_raises(self, monkeypatch):
        monkeypatch.setattr(pd.platform, "system", lambda: "OS/2")
        with pytest.raises(pd.UnsupportedPlatformError):
            pd.detect_os()


class TestDetectArch:
    @pytest.mark.parametrize(
        ("machine_value", "expected"),
        [("x86_64", "x64"), ("AMD64", "x64"), ("arm64", "arm64"), ("aarch64", "arm64")],
    )
    def test_maps_known_arches(self, monkeypatch, machine_value, expected):
        monkeypatch.setattr(pd.platform, "machine", lambda: machine_value)
        assert pd.detect_arch() == expected

    def test_unknown_arch_raises(self, monkeypatch):
        monkeypatch.setattr(pd.platform, "machine", lambda: "riscv64")
        with pytest.raises(pd.UnsupportedPlatformError):
            pd.detect_arch()


class TestDetectGpuBackend:
    def test_explicit_request_passes_through_unprobed(self, monkeypatch):
        # An explicit override must never be second-guessed by probing -- even if nvidia-smi is
        # absent, "cuda" should come back as "cuda" so the caller's asset-manifest lookup fails
        # loudly on a bad manifest key rather than silently falling back to a different backend.
        monkeypatch.setattr(pd, "_has_cmd", lambda name: False)
        assert pd.detect_gpu_backend("linux", "x64", "cuda") == "cuda"

    def test_apple_silicon_mac_is_metal(self):
        # Upstream's one macos-arm64 asset is the Metal build -- nothing to probe, but the backend
        # says what the binary uses rather than "cpu".
        assert pd.detect_gpu_backend("macos", "arm64", "auto") == "metal"

    def test_intel_mac_stays_cpu(self):
        assert pd.detect_gpu_backend("macos", "x64", "auto") == "cpu"

    def test_windows_is_always_cpu_for_now(self):
        assert pd.detect_gpu_backend("windows", "x64", "auto") == "cpu"

    def test_linux_detects_nvidia(self, monkeypatch):
        monkeypatch.setattr(pd, "_has_cmd", lambda name: name == "nvidia-smi")
        monkeypatch.setattr(pd, "_cmd_ok", lambda args: args[0] == "nvidia-smi")
        assert pd.detect_gpu_backend("linux", "x64", "auto") == "cuda"

    def test_linux_detects_amd_when_no_nvidia(self, monkeypatch):
        monkeypatch.setattr(pd, "_has_cmd", lambda name: name == "rocminfo")
        monkeypatch.setattr(pd, "_cmd_ok", lambda args: args[0] == "rocminfo")
        assert pd.detect_gpu_backend("linux", "x64", "auto") == "rocm"

    def test_linux_falls_back_to_cpu_when_neither_present(self, monkeypatch):
        monkeypatch.setattr(pd, "_has_cmd", lambda name: False)
        monkeypatch.setattr(pd, "_cmd_ok", lambda args: False)
        assert pd.detect_gpu_backend("linux", "x64", "auto") == "cpu"

    def test_linux_nvidia_present_but_not_working_falls_back(self, monkeypatch):
        # nvidia-smi on PATH but failing (e.g. driver/container mismatch) must not be treated as
        # "cuda available" -- a wrong positive here would hand a cuda asset to a host that can't
        # run it.
        monkeypatch.setattr(pd, "_has_cmd", lambda name: name == "nvidia-smi")
        monkeypatch.setattr(pd, "_cmd_ok", lambda args: False)
        assert pd.detect_gpu_backend("linux", "x64", "auto") == "cpu"


class TestDetectGlibcVersion:
    def test_parses_major_minor(self, monkeypatch):
        monkeypatch.setattr(pd.platform, "libc_ver", lambda: ("glibc", "2.35"))
        assert pd.detect_glibc_version() == (2, 35)

    def test_parses_a_patch_suffix(self, monkeypatch):
        # Some distros report a third component; only major.minor is ever compared.
        monkeypatch.setattr(pd.platform, "libc_ver", lambda: ("glibc", "2.38.1"))
        assert pd.detect_glibc_version() == (2, 38)

    def test_non_glibc_libc_is_unknown(self, monkeypatch):
        # macOS/Windows/musl all report something other than "glibc" here.
        monkeypatch.setattr(pd.platform, "libc_ver", lambda: ("", ""))
        assert pd.detect_glibc_version() is None

    def test_unparseable_version_is_unknown_not_a_crash(self, monkeypatch):
        monkeypatch.setattr(pd.platform, "libc_ver", lambda: ("glibc", "weird"))
        assert pd.detect_glibc_version() is None


class TestGlibcSatisfies:
    def test_no_floor_recorded_is_always_satisfied(self):
        assert pd.glibc_satisfies(None, (2, 30)) is True

    def test_unknown_host_is_never_blocked(self):
        # We can't prove the host is too old, so don't force a source build off a guess.
        assert pd.glibc_satisfies("2.38", None) is True

    def test_host_meets_floor_exactly(self):
        assert pd.glibc_satisfies("2.38", (2, 38)) is True

    def test_host_exceeds_floor(self):
        assert pd.glibc_satisfies("2.34", (2, 39)) is True

    def test_host_below_floor_fails(self):
        # The confirmed real case: JetPack 6.2.3's glibc 2.35 against the b10989 arm64 asset's 2.38.
        assert pd.glibc_satisfies("2.38", (2, 35)) is False

    def test_major_version_dominates_minor(self):
        assert pd.glibc_satisfies("2.38", (3, 0)) is True
        assert pd.glibc_satisfies("3.5", (2, 40)) is False


class TestDetectCudaComputeCapability:
    def test_no_nvidia_smi_is_unknown(self, monkeypatch):
        monkeypatch.setattr(pd, "_has_cmd", lambda name: False)
        assert pd.detect_cuda_compute_capability() is None

    def test_parses_real_csv_noheader_output(self, monkeypatch):
        monkeypatch.setattr(pd, "_has_cmd", lambda name: name == "nvidia-smi")

        class _Result:
            returncode = 0
            stdout = "8.7, Orin (nvgpu)\n"

        monkeypatch.setattr(pd.subprocess, "run", lambda *a, **k: _Result())
        assert pd.detect_cuda_compute_capability() == "8.7"

    def test_nonzero_exit_is_unknown(self, monkeypatch):
        monkeypatch.setattr(pd, "_has_cmd", lambda name: name == "nvidia-smi")

        class _Result:
            returncode = 1
            stdout = ""

        monkeypatch.setattr(pd.subprocess, "run", lambda *a, **k: _Result())
        assert pd.detect_cuda_compute_capability() is None

    def test_timeout_is_unknown_not_a_crash(self, monkeypatch):
        monkeypatch.setattr(pd, "_has_cmd", lambda name: name == "nvidia-smi")

        def _raise(*a, **k):
            raise pd.subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)

        monkeypatch.setattr(pd.subprocess, "run", _raise)
        assert pd.detect_cuda_compute_capability() is None

    def test_unparseable_output_is_unknown(self, monkeypatch):
        monkeypatch.setattr(pd, "_has_cmd", lambda name: name == "nvidia-smi")

        class _Result:
            returncode = 0
            stdout = "[N/A]\n"

        monkeypatch.setattr(pd.subprocess, "run", lambda *a, **k: _Result())
        assert pd.detect_cuda_compute_capability() is None


class TestDetectHostProfile:
    def test_combines_os_arch_backend(self, monkeypatch):
        monkeypatch.setattr(pd, "detect_os", lambda: "linux")
        monkeypatch.setattr(pd, "detect_arch", lambda: "x64")
        monkeypatch.setattr(pd, "detect_gpu_backend", lambda os_name, arch, requested: "cpu")
        profile = pd.detect_host_profile("auto")
        assert profile == pd.HostProfile(os_name="linux", arch="x64", backend="cpu")

    def test_asset_key_property(self):
        profile = pd.HostProfile(os_name="linux", arch="arm64", backend="cuda")
        assert profile.asset_key == "linux-arm64-cuda"
