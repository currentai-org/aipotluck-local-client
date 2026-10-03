"""aipotluck.diagnostics -- the GET /capabilities system/capability fingerprint.

Every detection primitive is monkeypatched (these are already covered for real by
test_platform_detect.py/test_source_build.py/test_build_strategy.py/test_build_cache.py) -- what
this file pins is (1) the RESPONSE SHAPE, and (2) that one probe failing degrades only its own
section rather than taking the whole thing down, since that resilience is this endpoint's entire
reason to exist (it has to stay useful on a host that's already in a broken state).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aipotluck import diagnostics as diag
from aipotluck.installer.platform_detect import HostProfile


@pytest.fixture(autouse=True)
def _stable_environment(monkeypatch):
    """A fully-known-good Linux/x64/CPU host, so tests only need to override what they're
    actually exercising."""
    monkeypatch.setattr(diag, "detect_os", lambda: "linux")
    monkeypatch.setattr(diag, "detect_arch", lambda: "x64")
    monkeypatch.setattr(diag, "detect_gpu_backend", lambda os_name, arch, requested: "cpu")
    monkeypatch.setattr(diag, "detect_glibc_version", lambda: (2, 39))
    monkeypatch.setattr(diag, "detect_cuda_compute_capability", lambda: None)
    monkeypatch.setattr(diag.platform, "system", lambda: "Linux")
    monkeypatch.setattr(diag.platform, "release", lambda: "6.8.0")
    monkeypatch.setattr(diag.platform, "version", lambda: "#1 SMP")
    monkeypatch.setattr(diag.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(diag.platform, "libc_ver", lambda: ("glibc", "2.39"))
    monkeypatch.setattr(diag.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(diag.source_build, "_find_cxx_compiler", lambda: "/usr/bin/g++")
    monkeypatch.setattr(diag.source_build, "_has_openssl_headers", lambda: True)
    monkeypatch.setattr(diag.source_build, "_has_vulkan_headers", lambda: False)
    monkeypatch.setattr(diag.source_build, "find_nvcc", lambda: None)
    monkeypatch.setattr(diag.source_build, "find_glslc", lambda: None)
    monkeypatch.setattr(diag.source_build, "find_hipcc", lambda: None)
    monkeypatch.setattr(diag.source_build, "missing_apt_packages", lambda: [])
    monkeypatch.setattr(diag.source_build, "_available_memory_gb", lambda: 8.0)
    monkeypatch.setattr(diag.source_build, "check_free_disk_gb", lambda path, **kw: 100.0)
    monkeypatch.setattr(diag.ctypes.util, "find_library", lambda name: "libgomp.so.1")
    monkeypatch.setattr(
        diag.fetch, "load_version_manifest",
        lambda path: {"tag": "b10989", "assets": {"linux-x64-cpu": {"file": "x.tar.gz"}}},
    )
    monkeypatch.setattr(
        diag.build_strategy, "resolve_install_strategy",
        lambda profile, manifest: diag.build_strategy.InstallStrategy(
            use_source_build=False, reason="ok", asset_key="linux-x64-cpu", asset_entry={"file": "x.tar.gz"}
        ),
    )
    monkeypatch.setattr(diag.build_cache, "load_custom_manifest", lambda path: {"assets": {}})


class TestRuntimeParams:
    """The traceability surface CLAUDE.md's "Runtime parameters" convention requires: any
    automatically-computed runtime parameter (per-model ctx_size/parallel/cache_type_k/-v, per
    CUR-1965's router-mode auto-sizing) must be readable from one real place, not reconstructed
    per-consumer. Router mode means there's no single "active model" anymore -- this reports
    router-level params plus every SIZED model, keyed by id, read from real files (the
    --models-preset INI + its sibling tuning JSON) rather than pure dict logic, so real file I/O
    is exercised here too (not mocked -- model_presets.py's own module already has that job)."""

    def test_extracts_router_level_fields(self):
        params = diag.runtime_params(
            {"llama_cpp": {"host": "127.0.0.1", "port": 8080, "gpu_layers": "auto", "models_max": 1}}
        )
        assert params["host"] == "127.0.0.1"
        assert params["port"] == 8080
        assert params["gpu_layers"] == "auto"
        assert params["models_max"] == 1

    def test_missing_llama_cpp_section_is_all_nones_not_a_crash(self):
        params = diag.runtime_params({})
        assert params["host"] is None
        assert params["models"] == {}

    def test_no_presets_path_means_empty_models_not_a_crash(self):
        params = diag.runtime_params({"llama_cpp": {"host": "127.0.0.1"}})
        assert params["models"] == {}

    def test_every_sized_model_is_reported_with_its_params_and_tuning(self, tmp_path):
        from aipotluck.installer import model_presets

        presets_path = tmp_path / "presets.ini"
        model_presets.write_preset(
            presets_path, "org/a:Q4_K_M",
            {"ctx-size": "16384", "parallel": "1", "cache-type-k": "q8_0", "cache-type-v": "q8_0"},
        )
        model_presets.write_tuning(presets_path, "org/a:Q4_K_M", {"ctx_size": "largest context that fits"})

        params = diag.runtime_params({"llama_cpp": {"presets_path": str(presets_path)}})

        assert params["models"] == {
            "org/a:Q4_K_M": {
                "ctx_size": "16384", "parallel": "1", "cache_type_k": "q8_0", "cache_type_v": "q8_0",
                "tuning": {"ctx_size": "largest context that fits"},
                "performance": None,  # never measured
            }
        }

    def test_a_measured_model_surfaces_its_localscore(self, tmp_path):
        """A score decides whether a model is installable, so it has to be inspectable afterwards
        -- the same traceability rule the sizing parameters follow."""
        from aipotluck.installer import model_localscore, model_perf_store, model_presets

        presets_path = tmp_path / "presets.ini"
        preset = {"ctx-size": "16384"}
        model_presets.write_preset(presets_path, "org/a:Q4_K_M", preset)
        cost = model_localscore.CostModel({256: 1.0, 1024: 1.0, 2048: 1.0}, 20.0, 0.0)
        result = model_localscore.localscore(cost)
        llama_cfg = {"presets_path": str(presets_path), "tag": "b10989"}
        model_perf_store.write_record(presets_path, "org/a:Q4_K_M", result,
                                      llama_config=llama_cfg, preset_args=preset,
                                      engine="b10989", ctx_size=16384)

        perf = diag.runtime_params({"llama_cpp": llama_cfg})["models"]["org/a:Q4_K_M"]["performance"]
        # the store rounds to one decimal on purpose -- a score is not precise to more than that
        assert perf["localscore"] == pytest.approx(result.score, abs=0.05)
        assert perf["band"] == result.band
        assert perf["engine"] == "b10989"
        assert perf["stale"] is False

    def test_a_model_with_no_tuning_entry_gets_an_empty_dict_not_a_crash(self, tmp_path):
        from aipotluck.installer import model_presets

        presets_path = tmp_path / "presets.ini"
        model_presets.write_preset(presets_path, "org/a:Q4_K_M", {"ctx-size": "4096"})

        params = diag.runtime_params({"llama_cpp": {"presets_path": str(presets_path)}})

        assert params["models"]["org/a:Q4_K_M"]["tuning"] == {}

    def test_embedded_in_current_install_info(self, tmp_path):
        from aipotluck.installer import model_presets

        presets_path = tmp_path / "presets.ini"
        model_presets.write_preset(presets_path, "org/a:Q4_K_M", {"ctx-size": "32768"})
        model_presets.write_tuning(presets_path, "org/a:Q4_K_M", {"ctx_size": "auto: test reason"})

        fp = diag.gather_fingerprint(config_dir=None, runtime_config={"llama_cpp": {"presets_path": str(presets_path)}})

        assert fp["current_install"]["runtime_params"]["models"]["org/a:Q4_K_M"]["ctx_size"] == "32768"
        assert fp["current_install"]["runtime_params"]["models"]["org/a:Q4_K_M"]["tuning"] == {
            "ctx_size": "auto: test reason"
        }


class TestGatherFingerprintShape:
    def test_top_level_sections_present(self):
        fp = diag.gather_fingerprint(config_dir=None, runtime_config=None)
        for key in (
            "service", "fingerprint_version", "os", "arch", "libc", "cpu", "disk", "gpu",
            "build_tools", "install_strategy", "current_install",
        ):
            assert key in fp, key

    def test_os_and_arch_reflect_detection(self):
        fp = diag.gather_fingerprint(config_dir=None, runtime_config=None)
        assert fp["os"]["name"] == "linux"
        assert fp["arch"]["normalized"] == "x64"

    def test_libc_reports_parsed_tuple(self):
        fp = diag.gather_fingerprint(config_dir=None, runtime_config=None)
        assert fp["libc"]["version_tuple"] == [2, 39]

    def test_build_tools_reflect_probes(self):
        fp = diag.gather_fingerprint(config_dir=None, runtime_config=None)
        assert fp["build_tools"]["cmake"] is True
        assert fp["build_tools"]["openssl_headers"] is True
        assert fp["build_tools"]["nvcc"] is None

    def test_install_strategy_reflects_resolved_strategy(self):
        fp = diag.gather_fingerprint(config_dir=None, runtime_config=None)
        assert fp["install_strategy"]["use_source_build"] is False
        assert fp["install_strategy"]["upstream_asset_key"] == "linux-x64-cpu"
        assert fp["install_strategy"]["pinned_tag"] == "b10989"

    def test_current_install_reflects_runtime_config(self, tmp_path):
        binary = tmp_path / "llama-server"
        binary.write_text("x")
        runtime_config = {
            "logged_in": True,
            "llama_cpp": {
                "tag": "b10989", "asset_key": "linux-x64-cpu", "built_from_source": False,
                "server_binary": str(binary),
            },
        }
        fp = diag.gather_fingerprint(config_dir=None, runtime_config=runtime_config)
        assert fp["current_install"]["logged_in"] is True
        assert fp["current_install"]["server_binary_exists"] is True

    def test_missing_server_binary_is_reported_false_not_a_crash(self, tmp_path):
        runtime_config = {
            "logged_in": True,
            "llama_cpp": {"server_binary": str(tmp_path / "nope")},
        }
        fp = diag.gather_fingerprint(config_dir=None, runtime_config=runtime_config)
        assert fp["current_install"]["server_binary_exists"] is False

    def test_no_runtime_config_is_handled(self):
        fp = diag.gather_fingerprint(config_dir=None, runtime_config=None)
        assert fp["current_install"]["logged_in"] is False
        assert fp["current_install"]["server_binary"] is None


class TestGatherFingerprintResilience:
    def test_one_failing_probe_does_not_take_down_the_response(self, monkeypatch):
        def _boom():
            raise RuntimeError("simulated probe failure")

        monkeypatch.setattr(diag, "detect_cuda_compute_capability", _boom)
        fp = diag.gather_fingerprint(config_dir=None, runtime_config=None)
        assert "error" in fp["gpu"]
        # Everything else still reports normally.
        assert fp["os"]["name"] == "linux"
        assert fp["build_tools"]["cmake"] is True

    def test_unsupported_os_still_returns_a_usable_response(self, monkeypatch):
        monkeypatch.setattr(diag, "detect_os", lambda: (_ for _ in ()).throw(diag.UnsupportedPlatformError("x")))
        fp = diag.gather_fingerprint(config_dir=None, runtime_config=None)
        assert fp["os"]["name"] is None
        assert "detect_error" in fp["os"]
        assert fp["install_strategy"]["error"]

    def test_missing_manifest_is_reported_not_raised(self, monkeypatch):
        monkeypatch.setattr(diag, "REPO_ROOT", Path("/definitely/does/not/exist"))
        fp = diag.gather_fingerprint(config_dir=None, runtime_config=None)
        assert "error" in fp["install_strategy"]


class TestGatherFingerprintCustomCache:
    def test_cache_hit_is_reported(self, monkeypatch):
        monkeypatch.setattr(diag, "detect_gpu_backend", lambda os_name, arch, requested: "cuda")
        monkeypatch.setattr(diag, "detect_cuda_compute_capability", lambda: "8.7")
        monkeypatch.setattr(
            diag.build_strategy, "resolve_install_strategy",
            lambda profile, manifest: diag.build_strategy.InstallStrategy(
                use_source_build=True, reason="no asset", cuda_arch="87"
            ),
        )
        monkeypatch.setattr(
            diag.build_cache, "load_custom_manifest",
            lambda path: {"assets": {"linux-x64-cuda-sm87": {"file": "f.tar.gz", "sha256": "abc"}}},
        )
        fp = diag.gather_fingerprint(config_dir=None, runtime_config=None)
        assert fp["install_strategy"]["custom_cache_available"] is True
        assert fp["install_strategy"]["custom_cache_key"] == "linux-x64-cuda-sm87"
        # A cache hit means no real build would happen -- recommended_build is only meaningful
        # when a real compile is actually what would happen.
        assert fp["install_strategy"]["recommended_build"] is None

    def test_source_build_without_cache_reports_recommended_build(self, monkeypatch):
        monkeypatch.setattr(diag, "detect_gpu_backend", lambda os_name, arch, requested: "cuda")
        monkeypatch.setattr(diag, "detect_cuda_compute_capability", lambda: "87")
        monkeypatch.setattr(
            diag.build_strategy, "resolve_install_strategy",
            lambda profile, manifest: diag.build_strategy.InstallStrategy(
                use_source_build=True, reason="no asset", cuda_arch="87"
            ),
        )
        fp = diag.gather_fingerprint(config_dir=None, runtime_config=None)
        assert fp["install_strategy"]["recommended_build"]["cmake_cuda_architectures"] == "87"
