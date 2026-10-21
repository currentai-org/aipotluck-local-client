"""aipotluck.installer.fetch -- resolving a binary inside an extracted llama.cpp archive.

These exist because of a live failure on a Jetson Orin NX: a leftover `llama-b10989.bak-<epoch>`
directory sat beside the real extraction, `find_binary` returned whichever copy the filesystem
happened to yield first, and `server_binary` in runtime.json ended up naming the stale one. The
install kept serving (the stale copy has a working llama-server), so nothing complained -- the only
visible symptom was the capability check reporting llama-bench missing, because it looks for
llama-bench beside llama-server and the stale copy predates it.
"""

from __future__ import annotations

import pytest

from aipotluck.installer import fetch
from aipotluck.installer.platform_detect import HostProfile


def _make_binary(directory, stem="llama-server"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / stem
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


class TestFindBinary:
    def test_raises_when_the_binary_is_genuinely_absent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            fetch.find_binary(tmp_path, "llama-server")

    def test_finds_a_binary_nested_one_folder_deep(self, tmp_path):
        expected = _make_binary(tmp_path / "llama-b10989")
        assert fetch.find_binary(tmp_path, "llama-server") == expected

    def test_prefers_the_real_extraction_over_a_bak_copy_beside_it(self, tmp_path):
        """The Jetson case. Both directories hold a working llama-server, so the only thing that
        can distinguish them is the name -- and picking the stale one is silent."""
        real = _make_binary(tmp_path / "llama-b10989")
        _make_binary(tmp_path / "llama-b10989.bak-1790802474")
        assert fetch.find_binary(tmp_path, "llama-server") == real

    def test_prefers_the_real_extraction_whichever_order_the_filesystem_yields(
        self, tmp_path, monkeypatch
    ):
        """Pairs with the test above to make the guard order-proof. The old code had no opinion
        about which copy it liked -- it inherited rglob's arbitrary order -- so on any given
        filesystem exactly one of these two directions picks the stale copy, and which one is not
        knowable in advance. Either test alone could pass by luck; together they cannot."""
        real = _make_binary(tmp_path / "llama-b10989")
        _make_binary(tmp_path / "llama-b10989.bak-1790802474")

        real_rglob = type(tmp_path).rglob
        monkeypatch.setattr(
            type(tmp_path), "rglob", lambda self, pat: list(real_rglob(self, pat))[::-1]
        )
        assert fetch.find_binary(tmp_path, "llama-server") == real

    def test_prefers_the_shallower_copy(self, tmp_path):
        shallow = _make_binary(tmp_path / "llama-b10989")
        _make_binary(tmp_path / "llama-b10989" / "nested" / "deeper")
        assert fetch.find_binary(tmp_path, "llama-server") == shallow

    def test_still_resolves_when_every_copy_looks_like_debris(self, tmp_path):
        """The debris heuristic must never be the reason an install finds nothing: a user whose
        only extraction happens to sit under an unluckily-named directory still gets a binary."""
        only = _make_binary(tmp_path / "llama-b10989.bak-1790802474")
        assert fetch.find_binary(tmp_path, "llama-server") == only

    def test_warns_when_more_than_one_copy_exists(self, tmp_path, caplog):
        """Ambiguity is reported, not silently resolved -- a wrong pick leaves a working install
        pointed at the wrong engine, which is exactly the failure that went unnoticed on the
        Jetson."""
        _make_binary(tmp_path / "llama-b10989")
        _make_binary(tmp_path / "llama-b10989.bak-1790802474")
        with caplog.at_level("WARNING"):
            fetch.find_binary(tmp_path, "llama-server")
        assert "Found 2 copies of llama-server" in caplog.text

    def test_does_not_warn_for_an_unambiguous_install(self, tmp_path, caplog):
        _make_binary(tmp_path / "llama-b10989")
        with caplog.at_level("WARNING"):
            fetch.find_binary(tmp_path, "llama-server")
        assert caplog.text == ""

    def test_resolves_llama_bench_the_same_way(self, tmp_path):
        """find_binary is generic over the stem, and the capability check depends on that."""
        real = _make_binary(tmp_path / "llama-b10989", stem="llama-bench")
        _make_binary(tmp_path / "llama-b10989.bak-1790802474", stem="llama-bench")
        assert fetch.find_binary(tmp_path, "llama-bench") == real


class TestResolveAssetAliases:
    MANIFEST = {
        "assets": {
            "macos-arm64-metal": {"file": "macos-arm64.tar.gz"},
            "linux-x64-cpu": {"file": "x64.tar.gz"},
        },
        "aliases": {"_comment": "ignored", "macos-arm64-cpu": "macos-arm64-metal", "linux-x64-cuda": "missing"},
    }

    def test_an_aliased_key_resolves_to_its_target_and_returns_the_targets_own_key(self):
        profile = HostProfile(os_name="macos", arch="arm64", backend="cpu")
        assert fetch.resolve_asset(self.MANIFEST, profile) == ("macos-arm64-metal", {"file": "macos-arm64.tar.gz"})

    def test_an_alias_whose_target_is_missing_is_ignored_and_the_cpu_fallback_still_applies(self):
        profile = HostProfile(os_name="linux", arch="x64", backend="cuda")
        assert fetch.resolve_asset(self.MANIFEST, profile)[0] == "linux-x64-cpu"

    def test_a_manifest_with_no_aliases_section_behaves_as_before(self):
        manifest = {"assets": {"linux-x64-cpu": {"file": "x64.tar.gz"}}}
        profile = HostProfile(os_name="linux", arch="x64", backend="cpu")
        assert fetch.resolve_asset(manifest, profile)[0] == "linux-x64-cpu"



class TestFetchCompanions:
    """Windows CUDA: llama.cpp ships the CUDA runtime DLLs as a separate cudart zip that must land
    beside llama-server, or the CUDA backend quietly fails to load and everything runs on the CPU."""

    def _zip(self, path, names):
        import zipfile as zf
        with zf.ZipFile(path, "w") as z:
            for n in names:
                z.writestr(n, b"x")
        return path

    def test_unpacks_each_companion_beside_the_binaries_once(self, tmp_path, monkeypatch):
        llama_dir = tmp_path / "llama"
        llama_dir.mkdir()
        archive = self._zip(tmp_path / "cudart.zip", ["cudart64_12.dll", "cublas64_12.dll"])
        calls = []

        def fake_download(manifest, entry, dest):
            calls.append(entry["file"])
            return archive

        monkeypatch.setattr(fetch, "download_asset", fake_download)
        entry = {"file": "llama-win-cuda.zip", "companions": [{"file": "cudart.zip", "sha256": "0" * 64}]}
        fetch.fetch_companions({}, entry, tmp_path / "dl", llama_dir)
        assert (llama_dir / "cudart64_12.dll").exists() and (llama_dir / "cublas64_12.dll").exists()
        fetch.fetch_companions({}, entry, tmp_path / "dl", llama_dir)
        assert calls == ["cudart.zip"]  # second run is a no-op

    def test_an_asset_without_companions_downloads_nothing_extra(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fetch, "download_asset", lambda *a: pytest.fail("no download expected"))
        fetch.fetch_companions({}, {"file": "llama-win-vulkan.zip"}, tmp_path, tmp_path)

    def test_the_pinned_windows_cuda_asset_carries_its_cuda_runtime(self):
        import json
        from pathlib import Path
        manifest = json.loads((Path(__file__).resolve().parents[1] / "llama_version.json").read_text())
        cuda = manifest["assets"]["windows-x64-cuda"]
        assert cuda["cuda_version"] == "12.4"
        assert [c["file"] for c in cuda["companions"]] == ["cudart-llama-bin-win-cuda-12.4-x64.zip"]
        assert all(len(c["sha256"]) == 64 for c in cuda["companions"])
        assert "windows-x64-vulkan" in manifest["assets"]
