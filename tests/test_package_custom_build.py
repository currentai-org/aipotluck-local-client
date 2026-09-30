"""scripts/package_custom_build.py -- packaging a from-source build into the custom-cache format.

The relocatability check is deliberately NOT mocked (a real fake `llama-server` script, copied to
a real different path and actually executed) -- that check exists specifically because a static
"did cmake configure with the right flag" check would have missed the real bug this project hit
(CMAKE_INSTALL_RPATH=$ORIGIN omitted), so it has to be tested the same way it works: for real.
"""

from __future__ import annotations

import json
import stat
import tarfile
import textwrap
from pathlib import Path

import pytest

import package_custom_build as pcb
from aipotluck.installer.platform_detect import HostProfile

FAKE_LLAMA_SERVER_SCRIPT = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import os
    import sys

    if "--list-devices" in sys.argv:
        if os.environ.get("FAKE_RELOCATE_FAIL"):
            sys.stderr.write(
                "moved/llama-server: error while loading shared libraries: "
                "libfake.so.0: cannot open shared object file: No such file or directory\\n"
            )
            sys.exit(127)
        print("Available devices:\\n  CPU0: fake (n/a)")
        sys.exit(0)
    sys.exit(1)
    """
)


def _make_bin_dir(tmp_path: Path, *, with_bench: bool = False) -> Path:
    build_dir = tmp_path / "llama.cpp-build"
    bin_dir = build_dir / "bin"
    bin_dir.mkdir(parents=True)
    script = bin_dir / "llama-server"
    script.write_text(FAKE_LLAMA_SERVER_SCRIPT, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    # A companion shared lib + its usual versioned-symlink chain, matching what a real cmake
    # build actually produces (and what upstream's own release archives ship) -- proves staging
    # preserves symlinks rather than dereferencing them into duplicate real files.
    real_lib = bin_dir / "libggml-base.so.0.24.0"
    real_lib.write_text("fake shared lib contents", encoding="utf-8")
    (bin_dir / "libggml-base.so.0").symlink_to(real_lib.name)
    (bin_dir / "libggml-base.so").symlink_to("libggml-base.so.0")
    if with_bench:
        bench = bin_dir / "llama-bench"
        bench.write_text(FAKE_LLAMA_SERVER_SCRIPT, encoding="utf-8")
        bench.chmod(bench.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return build_dir


class TestResolveTag:
    def test_explicit_tag_wins(self):
        assert pcb.resolve_tag("custom-tag") == "custom-tag"

    def test_falls_back_to_pinned_manifest(self, monkeypatch):
        monkeypatch.setattr(pcb.fetch, "load_version_manifest", lambda path: {"tag": "b10989"})
        assert pcb.resolve_tag(None) == "b10989"


class TestResolveKey:
    def test_explicit_key_wins(self):
        assert pcb.resolve_key("some-explicit-key") == "some-explicit-key"

    def test_auto_derives_cpu_key(self, monkeypatch):
        monkeypatch.setattr(pcb, "detect_host_profile", lambda backend: HostProfile("linux", "arm64", "cpu"))
        assert pcb.resolve_key(None) == "linux-arm64-cpu"

    def test_auto_derives_cuda_key_with_sm_suffix(self, monkeypatch):
        monkeypatch.setattr(pcb, "detect_host_profile", lambda backend: HostProfile("linux", "arm64", "cuda"))
        monkeypatch.setattr(pcb, "detect_cuda_compute_capability", lambda: "8.7")
        assert pcb.resolve_key(None) == "linux-arm64-cuda-sm87"

    def test_cuda_with_undetectable_arch_refuses_rather_than_guessing(self, monkeypatch):
        monkeypatch.setattr(pcb, "detect_host_profile", lambda backend: HostProfile("linux", "arm64", "cuda"))
        monkeypatch.setattr(pcb, "detect_cuda_compute_capability", lambda: None)
        with pytest.raises(pcb.PackagingError, match="compute capability"):
            pcb.resolve_key(None)


class TestStage:
    def test_missing_bin_dir_raises(self, tmp_path):
        with pytest.raises(pcb.PackagingError, match="No bin/ directory"):
            pcb.stage(tmp_path / "nonexistent-build", "b10989")

    def test_stages_into_a_flat_llama_tag_directory(self, tmp_path):
        build_dir = _make_bin_dir(tmp_path)
        staged = pcb.stage(build_dir, "b10989")
        try:
            assert staged.name == "llama-b10989"
            assert (staged / "llama-server").exists()
        finally:
            import shutil
            shutil.rmtree(staged.parent, ignore_errors=True)

    def test_preserves_symlinks_rather_than_dereferencing(self, tmp_path):
        build_dir = _make_bin_dir(tmp_path)
        staged = pcb.stage(build_dir, "b10989")
        try:
            assert (staged / "libggml-base.so").is_symlink()
            assert (staged / "libggml-base.so.0").is_symlink()
        finally:
            import shutil
            shutil.rmtree(staged.parent, ignore_errors=True)


class TestVerifyRelocatable:
    def test_a_real_relocatable_build_passes(self, tmp_path):
        build_dir = _make_bin_dir(tmp_path)
        staged = pcb.stage(build_dir, "b10989")
        pcb.verify_relocatable(staged)  # must not raise

    def test_a_non_relocatable_build_is_caught_by_its_real_failure_signature(self, tmp_path, monkeypatch):
        # This is the actual bug this project shipped once: a binary that only works from the
        # exact path it was built at fails with a dynamic-linker error, not a clean non-zero exit.
        monkeypatch.setenv("FAKE_RELOCATE_FAIL", "1")
        build_dir = _make_bin_dir(tmp_path)
        staged = pcb.stage(build_dir, "b10989")
        with pytest.raises(pcb.PackagingError, match="not relocatable"):
            pcb.verify_relocatable(staged)

    def test_missing_binary_after_staging_is_a_clean_error(self, tmp_path):
        staged = tmp_path / "llama-b10989"
        staged.mkdir()
        with pytest.raises(pcb.PackagingError):
            pcb.verify_relocatable(staged)

    def test_llama_bench_is_verified_too_when_the_build_has_one(self, tmp_path, monkeypatch, caplog):
        """llama-bench is what model_perf grades a model with, and like llama-server it is a thin
        launcher over its own *-impl.so -- the exact shape that breaks on a bad RPATH. Shipping a
        broken one would silently push every host onto the slower live-server fallback."""
        build_dir = _make_bin_dir(tmp_path, with_bench=True)
        staged = pcb.stage(build_dir, "b10989")
        with caplog.at_level("INFO"):
            pcb.verify_relocatable(staged)
        assert "llama-bench" in caplog.text

    def test_a_non_relocatable_llama_bench_fails_packaging(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FAKE_RELOCATE_FAIL", "1")
        build_dir = _make_bin_dir(tmp_path, with_bench=True)
        staged = pcb.stage(build_dir, "b10989")
        with pytest.raises(pcb.PackagingError, match="not relocatable"):
            pcb.verify_relocatable(staged)

    def test_a_build_without_llama_bench_still_packages_with_a_warning(self, tmp_path, caplog):
        """Older archives predate llama-bench being a build target; they must still package."""
        build_dir = _make_bin_dir(tmp_path, with_bench=False)
        staged = pcb.stage(build_dir, "b10989")
        with caplog.at_level("WARNING"):
            pcb.verify_relocatable(staged)
        assert "llama-bench is not in this build" in caplog.text


class TestPackage:
    def test_produces_a_tarball_matching_upstreams_flat_layout(self, tmp_path):
        build_dir = _make_bin_dir(tmp_path)
        staged = pcb.stage(build_dir, "b10989")
        out_dir = tmp_path / "out"
        archive = pcb.package(staged, "b10989", "linux-arm64-cuda-sm87", out_dir)

        assert archive.name == "llama-b10989-bin-linux-arm64-cuda-sm87.tar.gz"
        with tarfile.open(archive) as tf:
            names = tf.getnames()
        assert "llama-b10989" in names
        assert "llama-b10989/llama-server" in names
        # Never nested under bin/ -- matches upstream's own release archive layout exactly.
        assert not any("/bin/" in n for n in names)

    def test_sha256_of_is_correct(self, tmp_path):
        path = tmp_path / "f.txt"
        path.write_bytes(b"hello world")
        import hashlib
        assert pcb.sha256_of(path) == hashlib.sha256(b"hello world").hexdigest()


class TestFormatNextSteps:
    def test_includes_upload_command_and_manifest_snippet(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            pcb.build_cache, "load_custom_manifest",
            lambda path: {"tag": "custom-builds", "release_base_url": "https://example.test/dl"},
        )
        output = pcb.format_next_steps(tmp_path / "llama-b10989-bin-linux-arm64-cuda-sm87.tar.gz", "abc123", "b10989", "linux-arm64-cuda-sm87")
        assert "gh release upload custom-builds" in output
        assert "abc123" in output
        assert '"linux-arm64-cuda-sm87"' in output
        assert "https://example.test/dl/custom-builds/llama-b10989-bin-linux-arm64-cuda-sm87.tar.gz" in output


class TestMainEndToEnd:
    def test_happy_path_produces_a_real_archive_and_prints_next_steps(self, tmp_path, monkeypatch, capsys):
        build_dir = _make_bin_dir(tmp_path)
        out_dir = tmp_path / "out"
        monkeypatch.setattr(pcb.build_cache, "load_custom_manifest", lambda path: {"tag": "custom-builds"})

        rc = pcb.main([
            "--build-dir", str(build_dir), "--tag", "b10989", "--key", "linux-arm64-cuda-sm87",
            "--output-dir", str(out_dir),
        ])

        assert rc == 0
        archive = out_dir / "llama-b10989-bin-linux-arm64-cuda-sm87.tar.gz"
        assert archive.exists()
        out = capsys.readouterr().out
        assert "Packaged:" in out
        assert "sha256:" in out

    def test_skip_relocatability_check_flag_bypasses_it(self, tmp_path, monkeypatch, capsys):
        # Would otherwise fail the relocatability check; --skip-relocatability-check must let it
        # through anyway (with a warning, which is a log record, not asserted here).
        monkeypatch.setenv("FAKE_RELOCATE_FAIL", "1")
        build_dir = _make_bin_dir(tmp_path)
        out_dir = tmp_path / "out"
        monkeypatch.setattr(pcb.build_cache, "load_custom_manifest", lambda path: {"tag": "custom-builds"})

        rc = pcb.main([
            "--build-dir", str(build_dir), "--tag", "b10989", "--key", "linux-arm64-cuda-sm87",
            "--output-dir", str(out_dir), "--skip-relocatability-check",
        ])

        assert rc == 0
        assert (out_dir / "llama-b10989-bin-linux-arm64-cuda-sm87.tar.gz").exists()

    def test_non_relocatable_build_refuses_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FAKE_RELOCATE_FAIL", "1")
        build_dir = _make_bin_dir(tmp_path)
        out_dir = tmp_path / "out"

        rc = pcb.main([
            "--build-dir", str(build_dir), "--tag", "b10989", "--key", "linux-arm64-cuda-sm87",
            "--output-dir", str(out_dir),
        ])

        assert rc == 1
        assert not (out_dir / "llama-b10989-bin-linux-arm64-cuda-sm87.tar.gz").exists()

    def test_staging_dir_is_cleaned_up_even_on_failure(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FAKE_RELOCATE_FAIL", "1")
        build_dir = _make_bin_dir(tmp_path)
        staged_roots_before = set(Path(tempfile_dir()).glob("package_custom_build-*"))

        pcb.main([
            "--build-dir", str(build_dir), "--tag", "b10989", "--key", "linux-arm64-cuda-sm87",
            "--output-dir", str(tmp_path / "out"),
        ])

        staged_roots_after = set(Path(tempfile_dir()).glob("package_custom_build-*"))
        assert staged_roots_after == staged_roots_before


def tempfile_dir() -> str:
    import tempfile
    return tempfile.gettempdir()
