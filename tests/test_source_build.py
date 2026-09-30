"""aipotluck.installer.source_build -- the "configure + build llama-server from source" path.

Deliberately NOT mocked: subprocess orchestration, timeout handling, and process-group termination
are exactly the kind of thing a mock can make LOOK correct without proving it -- a fake `cmake`
stand-in script plays the part of the real tool instead, the same approach test_model_pull.py uses
for llama-server itself. It supports two invocations (configure: `-S <src> -B <build>`; build:
`--build <dir> --target llama-server -j <n>`) and is steered entirely through environment
variables (FAKE_CMAKE_*) so each test can shape its behavior without new script variants.
"""

from __future__ import annotations

import stat
import textwrap
import time
from pathlib import Path

import pytest

from aipotluck.installer import source_build as sb

FAKE_CMAKE_SCRIPT = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import os
    import subprocess
    import sys
    import time

    args = sys.argv[1:]

    if args[0] == "--build":
        build_dir = args[1]
        target_name = args[args.index("--target") + 1] if "--target" in args else "llama-server"
        delay = float(os.environ.get("FAKE_CMAKE_BUILD_DELAY", "0"))
        heartbeat = os.path.join(build_dir, "heartbeat.txt")
        log_path = os.path.join(build_dir, "build_log.txt")
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write("built:" + target_name + "\\n")
        if delay:
            # A real child process (not just this script sleeping) so a process-group kill has
            # something real below it to prove it reaches -- exactly the shape `cmake --build`
            # actually has (cmake -> make/ninja -> compiler workers).
            child = subprocess.Popen(["sleep", str(delay)])
            deadline = time.monotonic() + delay
            while time.monotonic() < deadline and child.poll() is None:
                with open(heartbeat, "a", encoding="utf-8") as fh:
                    fh.write(f"{time.monotonic()}\\n")
                time.sleep(0.2)
            child.wait()
        if os.environ.get("FAKE_CMAKE_BUILD_FAIL"):
            sys.exit(1)
        if os.environ.get("FAKE_CMAKE_BENCH_FAIL") and target_name == "llama-bench":
            sys.exit(1)
        bin_dir = os.path.join(build_dir, "bin")
        os.makedirs(bin_dir, exist_ok=True)
        target = os.path.join(bin_dir, target_name)
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("#!/bin/sh\\necho fake\\n")
        os.chmod(target, 0o755)
        sys.exit(0)

    # configure mode: -S <src> -B <build> -D...
    build_dir = args[args.index("-B") + 1]
    os.makedirs(build_dir, exist_ok=True)
    with open(os.path.join(build_dir, "configured.marker"), "w", encoding="utf-8") as fh:
        fh.write("\\n".join(args))
    if os.environ.get("FAKE_CMAKE_CONFIGURE_FAIL"):
        sys.exit(1)
    sys.exit(0)
    """
)


@pytest.fixture
def fake_cmake(tmp_path: Path) -> Path:
    script = tmp_path / "fake_cmake.py"
    script.write_text(FAKE_CMAKE_SCRIPT, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


class TestCheckBuildPrerequisites:
    def _mock_all_present(self, monkeypatch):
        monkeypatch.setattr(sb.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(sb, "_has_openssl_headers", lambda: True)
        monkeypatch.setattr(sb, "_has_libgomp", lambda: True)

    def test_all_present_is_no_problems(self, monkeypatch):
        self._mock_all_present(monkeypatch)
        assert sb.check_build_prerequisites(backend="cpu") == []

    def test_missing_cmake_is_reported_with_the_apt_fix(self, monkeypatch):
        self._mock_all_present(monkeypatch)
        monkeypatch.setattr(sb.shutil, "which", lambda name: None if name == "cmake" else f"/usr/bin/{name}")
        problems = sb.check_build_prerequisites(backend="cpu")
        assert any("cmake" in p and "apt-get install" in p for p in problems)

    def test_missing_openssl_headers_is_reported_even_though_it_wont_fail_the_build(self, monkeypatch):
        # The whole reason this check exists: a missing libssl-dev does NOT fail cmake configure,
        # so if we didn't check for it here, nothing would ever catch it before a wasted build.
        self._mock_all_present(monkeypatch)
        monkeypatch.setattr(sb, "_has_openssl_headers", lambda: False)
        problems = sb.check_build_prerequisites(backend="cpu")
        assert any("libssl-dev" in p for p in problems)

    def test_missing_libgomp_is_reported_even_though_it_wont_fail_the_build(self, monkeypatch):
        # Same shape as OpenSSL -- find_package(OpenMP) degrades silently too (build-log warning,
        # not a configure failure), so this must be caught the same way.
        self._mock_all_present(monkeypatch)
        monkeypatch.setattr(sb, "_has_libgomp", lambda: False)
        problems = sb.check_build_prerequisites(backend="cpu")
        assert any("libgomp" in p for p in problems)

    def test_cuda_backend_but_no_nvcc_is_reported(self, monkeypatch):
        self._mock_all_present(monkeypatch)
        monkeypatch.setattr(sb, "find_nvcc", lambda: None)
        problems = sb.check_build_prerequisites(backend="cuda")
        assert any("nvcc" in p for p in problems)

    def test_cpu_backend_never_checks_for_nvcc(self, monkeypatch):
        self._mock_all_present(monkeypatch)
        monkeypatch.setattr(sb, "find_nvcc", lambda: None)
        assert sb.check_build_prerequisites(backend="cpu") == []

    def test_vulkan_backend_missing_glslc_or_headers_is_reported(self, monkeypatch):
        self._mock_all_present(monkeypatch)
        monkeypatch.setattr(sb, "find_glslc", lambda: None)
        monkeypatch.setattr(sb, "_has_vulkan_headers", lambda: True)
        problems = sb.check_build_prerequisites(backend="vulkan")
        assert any("Vulkan" in p for p in problems)

    def test_vulkan_backend_satisfied_is_no_problem(self, monkeypatch):
        self._mock_all_present(monkeypatch)
        monkeypatch.setattr(sb, "find_glslc", lambda: "/usr/bin/glslc")
        monkeypatch.setattr(sb, "_has_vulkan_headers", lambda: True)
        assert sb.check_build_prerequisites(backend="vulkan") == []

    def test_non_vulkan_backend_never_checks_vulkan(self, monkeypatch):
        self._mock_all_present(monkeypatch)
        monkeypatch.setattr(sb, "find_glslc", lambda: None)
        monkeypatch.setattr(sb, "_has_vulkan_headers", lambda: False)
        assert sb.check_build_prerequisites(backend="cpu") == []

    def test_rocm_backend_missing_hipcc_is_reported(self, monkeypatch):
        self._mock_all_present(monkeypatch)
        monkeypatch.setattr(sb, "find_hipcc", lambda: None)
        problems = sb.check_build_prerequisites(backend="rocm")
        assert any("hipcc" in p for p in problems)

    def test_gpu_vendor_toolchain_gaps_are_never_in_missing_apt_packages(self, monkeypatch):
        # nvcc/glslc/hipcc gaps must never appear as apt-installable -- see missing_apt_packages'
        # docstring on why those stay instructions no matter how apt-install is invoked.
        self._mock_all_present(monkeypatch)
        monkeypatch.setattr(sb, "find_nvcc", lambda: None)
        monkeypatch.setattr(sb, "find_glslc", lambda: None)
        monkeypatch.setattr(sb, "find_hipcc", lambda: None)
        assert sb.missing_apt_packages() == []


class TestFindNvcc:
    def test_prefers_path(self, monkeypatch):
        monkeypatch.setattr(sb.shutil, "which", lambda name: "/usr/bin/nvcc")
        assert sb.find_nvcc() == Path("/usr/bin/nvcc")

    def test_falls_back_to_cuda_glob(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sb.shutil, "which", lambda name: None)
        cuda_bin = tmp_path / "cuda-12.6" / "bin"
        cuda_bin.mkdir(parents=True)
        (cuda_bin / "nvcc").write_text("x")
        monkeypatch.setattr(sb.Path, "glob", lambda self, pattern: iter([cuda_bin / "nvcc"]))
        assert sb.find_nvcc() == cuda_bin / "nvcc"

    def test_nothing_found_is_none(self, monkeypatch):
        monkeypatch.setattr(sb.shutil, "which", lambda name: None)
        monkeypatch.setattr(sb.Path, "glob", lambda self, pattern: iter([]))
        assert sb.find_nvcc() is None


class TestRecommendedJobs:
    def test_unknown_memory_falls_back_to_cpu_count(self, monkeypatch):
        monkeypatch.setattr(sb.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(sb, "_available_memory_gb", lambda: None)
        assert sb.recommended_jobs(want_cuda=False) == 8

    def test_caps_by_available_memory_for_cpu_build(self, monkeypatch):
        monkeypatch.setattr(sb.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(sb, "_available_memory_gb", lambda: 3.5)
        assert sb.recommended_jobs(want_cuda=False) == 3

    def test_cuda_build_gets_a_bigger_per_job_memory_budget(self, monkeypatch):
        # The confirmed real case: a Jetson Orin Nano, 6 cores, ~5.6GB available -- nvcc's heavier
        # footprint means fewer parallel jobs than a plain C++ build would get on the same host.
        monkeypatch.setattr(sb.os, "cpu_count", lambda: 6)
        monkeypatch.setattr(sb, "_available_memory_gb", lambda: 5.6)
        assert sb.recommended_jobs(want_cuda=True) == 2
        assert sb.recommended_jobs(want_cuda=False) == 5

    def test_never_recommends_zero(self, monkeypatch):
        monkeypatch.setattr(sb.os, "cpu_count", lambda: 4)
        monkeypatch.setattr(sb, "_available_memory_gb", lambda: 0.1)
        assert sb.recommended_jobs(want_cuda=True) >= 1


class TestEnsureSourceCheckedOut:
    def test_missing_git_checkout_raises_with_actionable_message(self, tmp_path):
        (tmp_path / "vendor" / "llama.cpp").mkdir(parents=True)
        with pytest.raises(sb.BuildError, match="submodule update --init"):
            sb.ensure_source_checked_out(tmp_path, "b10989")


class TestBuildLlamaServer:
    def test_happy_path_builds_and_locates_the_binary(self, tmp_path, fake_cmake):
        source_dir = tmp_path / "src"
        build_dir = tmp_path / "build"
        source_dir.mkdir()
        binary = sb.build_llama_server(
            source_dir, build_dir, fingerprint="sha123:cpu", cmake_binary=str(fake_cmake),
        )
        assert binary.name == "llama-server"
        assert binary.exists()

    def test_configure_sets_a_relocatable_origin_rpath(self, tmp_path, fake_cmake):
        # Without this, the binary only works from the exact absolute path it was built at --
        # fine for a normal in-place source build, but silently broken the moment bin/ is
        # archived and extracted somewhere else (a different install root, a different host
        # entirely via the custom binary cache).
        source_dir = tmp_path / "src"
        build_dir = tmp_path / "build"
        source_dir.mkdir()
        sb.build_llama_server(source_dir, build_dir, fingerprint="x", cmake_binary=str(fake_cmake))
        marker = (build_dir / "configured.marker").read_text(encoding="utf-8")
        assert "-DCMAKE_INSTALL_RPATH=$ORIGIN" in marker
        assert "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON" in marker

    def test_passes_cuda_flags_through_at_configure_time(self, tmp_path, fake_cmake):
        source_dir = tmp_path / "src"
        build_dir = tmp_path / "build"
        source_dir.mkdir()
        sb.build_llama_server(
            source_dir, build_dir, fingerprint="sha123:87", cuda_arch="87", cmake_binary=str(fake_cmake),
        )
        marker = (build_dir / "configured.marker").read_text(encoding="utf-8")
        assert "-DGGML_CUDA=ON" in marker
        assert "-DCMAKE_CUDA_ARCHITECTURES=87" in marker

    def test_cuda_build_passes_the_located_nvcc_path_to_cmake(self, tmp_path, fake_cmake, monkeypatch):
        # Regression: cmake's own CUDA-compiler detection only searches PATH. Confirmed live on
        # the reference Jetson that nvcc is real but not on PATH (find_nvcc()'s glob fallback is
        # the only reason it was found at all) -- without threading that path through explicitly,
        # configure fails with CMAKE_CUDA_COMPILER-NOTFOUND despite find_nvcc() succeeding.
        monkeypatch.setattr(sb, "find_nvcc", lambda: Path("/usr/local/cuda-12.6/bin/nvcc"))
        source_dir = tmp_path / "src"
        build_dir = tmp_path / "build"
        source_dir.mkdir()
        sb.build_llama_server(
            source_dir, build_dir, fingerprint="x", cuda_arch="87", cmake_binary=str(fake_cmake),
        )
        marker = (build_dir / "configured.marker").read_text(encoding="utf-8")
        assert "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc" in marker

    def test_cuda_build_with_undiscoverable_nvcc_omits_the_compiler_flag(self, tmp_path, fake_cmake, monkeypatch):
        # Shouldn't happen in practice (check_build_prerequisites gates this earlier), but must
        # never pass a bogus/empty -DCMAKE_CUDA_COMPILER= if find_nvcc() ever comes back None here.
        monkeypatch.setattr(sb, "find_nvcc", lambda: None)
        source_dir = tmp_path / "src"
        build_dir = tmp_path / "build"
        source_dir.mkdir()
        sb.build_llama_server(
            source_dir, build_dir, fingerprint="x", cuda_arch="87", cmake_binary=str(fake_cmake),
        )
        marker = (build_dir / "configured.marker").read_text(encoding="utf-8")
        assert "CMAKE_CUDA_COMPILER" not in marker

    def test_cpu_only_build_does_not_pass_cuda_flags(self, tmp_path, fake_cmake):
        source_dir = tmp_path / "src"
        build_dir = tmp_path / "build"
        source_dir.mkdir()
        sb.build_llama_server(source_dir, build_dir, fingerprint="sha123:cpu", cmake_binary=str(fake_cmake))
        marker = (build_dir / "configured.marker").read_text(encoding="utf-8")
        assert "GGML_CUDA" not in marker

    def test_matching_fingerprint_skips_rebuilding(self, tmp_path, fake_cmake):
        source_dir = tmp_path / "src"
        build_dir = tmp_path / "build"
        source_dir.mkdir()
        sb.build_llama_server(source_dir, build_dir, fingerprint="sha123:cpu", cmake_binary=str(fake_cmake))
        sb.build_llama_server(source_dir, build_dir, fingerprint="sha123:cpu", cmake_binary=str(fake_cmake))
        log_lines = (build_dir / "build_log.txt").read_text(encoding="utf-8").splitlines()
        assert log_lines.count("built:llama-server") == 1

    def test_changed_fingerprint_forces_a_rebuild(self, tmp_path, fake_cmake):
        source_dir = tmp_path / "src"
        build_dir = tmp_path / "build"
        source_dir.mkdir()
        sb.build_llama_server(source_dir, build_dir, fingerprint="sha123:cpu", cmake_binary=str(fake_cmake))
        sb.build_llama_server(source_dir, build_dir, fingerprint="sha456:87", cuda_arch="87", cmake_binary=str(fake_cmake))
        log_lines = (build_dir / "build_log.txt").read_text(encoding="utf-8").splitlines()
        assert log_lines.count("built:llama-server") == 2

    def test_builds_llama_bench_alongside_llama_server(self, tmp_path, fake_cmake):
        """model_perf grades a model with llama-bench, and a source build is exactly the slow
        hardware where an unusable model is most likely -- so it has to come out of the same build
        that produces llama-server, before _cleanup_intermediate_objects makes a later one cost a
        full rebuild."""
        source_dir = tmp_path / "src"
        build_dir = tmp_path / "build"
        source_dir.mkdir()
        sb.build_llama_server(source_dir, build_dir, fingerprint="sha123:cpu", cmake_binary=str(fake_cmake))
        log_lines = (build_dir / "build_log.txt").read_text(encoding="utf-8").splitlines()
        assert log_lines == ["built:llama-server", "built:llama-bench"]
        assert (build_dir / "bin" / "llama-bench").exists()

    def test_llama_bench_build_failure_does_not_fail_the_install(self, tmp_path, fake_cmake, monkeypatch, caplog):
        """A missing benchmark binary is a degraded grade (model_perf falls back to measuring
        through the running server), never a broken device -- so it must not take the install with
        it when some toolchain refuses to build it."""
        monkeypatch.setenv("FAKE_CMAKE_BENCH_FAIL", "1")
        source_dir = tmp_path / "src"
        build_dir = tmp_path / "build"
        source_dir.mkdir()
        with caplog.at_level("WARNING"):
            binary = sb.build_llama_server(
                source_dir, build_dir, fingerprint="sha123:cpu", cmake_binary=str(fake_cmake),
            )
        assert binary.exists() and binary.name == "llama-server"
        assert not (build_dir / "bin" / "llama-bench").exists()
        assert "Could not build llama-bench" in caplog.text

    def test_configure_failure_raises_with_stderr(self, tmp_path, fake_cmake, monkeypatch):
        monkeypatch.setenv("FAKE_CMAKE_CONFIGURE_FAIL", "1")
        source_dir = tmp_path / "src"
        build_dir = tmp_path / "build"
        source_dir.mkdir()
        with pytest.raises(sb.BuildError, match="failed"):
            sb.build_llama_server(source_dir, build_dir, fingerprint="x", cmake_binary=str(fake_cmake))

    def test_build_failure_raises(self, tmp_path, fake_cmake, monkeypatch):
        monkeypatch.setenv("FAKE_CMAKE_BUILD_FAIL", "1")
        source_dir = tmp_path / "src"
        build_dir = tmp_path / "build"
        source_dir.mkdir()
        with pytest.raises(sb.BuildError, match="Building llama-server failed"):
            sb.build_llama_server(source_dir, build_dir, fingerprint="x", cmake_binary=str(fake_cmake))

    def test_timeout_really_kills_the_whole_process_tree(self, tmp_path, fake_cmake, monkeypatch):
        # Not just "raises on timeout" (easy to fake) -- proves the underlying child process
        # (the fake build's own `sleep`, standing in for a real compiler worker) actually stops,
        # the same way test_model_pull.py reconnects to a port to prove real termination.
        monkeypatch.setenv("FAKE_CMAKE_BUILD_DELAY", "30")
        source_dir = tmp_path / "src"
        build_dir = tmp_path / "build"
        source_dir.mkdir()
        with pytest.raises(sb.BuildError, match="did not finish"):
            sb.build_llama_server(
                source_dir, build_dir, fingerprint="x", cmake_binary=str(fake_cmake), timeout=1.0,
            )
        heartbeat = build_dir / "heartbeat.txt"
        count_at_kill = len(heartbeat.read_text(encoding="utf-8").splitlines())
        time.sleep(1.5)
        count_after_wait = len(heartbeat.read_text(encoding="utf-8").splitlines())
        assert count_after_wait == count_at_kill

    def test_intermediate_cmakefiles_dirs_are_cleaned_up_after_success(self, tmp_path, fake_cmake):
        source_dir = tmp_path / "src"
        build_dir = tmp_path / "build"
        source_dir.mkdir()
        sb.build_llama_server(source_dir, build_dir, fingerprint="x", cmake_binary=str(fake_cmake))
        nested = build_dir / "some" / "CMakeFiles"
        nested.mkdir(parents=True)
        (nested / "leftover.o").write_text("x")
        sb._cleanup_intermediate_objects(build_dir)
        assert not nested.exists()
