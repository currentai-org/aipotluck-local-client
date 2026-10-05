"""aipotluck.installer.model_preflight -- is this model simply bigger than the machine?

The point of this check is that it runs before the download, so every test here works from a file
listing alone. Resolution has to agree with llama.cpp's own `-hf` downloader, or we would measure
one file and fetch another; the cases below pin the parts of that agreement that actually bite
(quant matching, sidecar exclusion, sharded models).
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from aipotluck.installer import model_preflight as mp


def _tree(*entries: tuple[str, int]) -> list[dict]:
    return [{"type": "file", "path": path, "size": size} for path, size in entries]


@pytest.fixture
def listing(monkeypatch):
    """Installs a fake repository listing; returns a setter so each test states its own."""

    def _set(entries):
        monkeypatch.setattr(mp, "_fetch_tree", lambda repo, timeout: entries)

    return _set


class TestSplitRepoTag:
    def test_splits_a_tagged_id(self):
        assert mp.split_repo_tag("org/repo:Q4_K_M") == ("org/repo", "Q4_K_M")

    def test_an_untagged_id_has_an_empty_tag(self):
        assert mp.split_repo_tag("org/repo") == ("org/repo", "")


class TestModelFileRecognition:
    @pytest.mark.parametrize(
        "path",
        ["mmproj-model-f16.gguf", "model-imatrix.gguf", "mtp-draft.gguf", "eagle3-head.gguf"],
    )
    def test_sidecars_are_not_the_model(self, path):
        """These get downloaded alongside a model but are not what has to fit, and counting one as
        the model would report a few hundred MB for a 40GB pull."""
        assert not mp._is_model_file(path)

    def test_a_plain_gguf_is_the_model(self):
        assert mp._is_model_file("Model-Q4_K_M.gguf")

    def test_a_non_gguf_file_is_not(self):
        assert not mp._is_model_file("README.md")


class TestRemoteModelBytes:
    def test_picks_the_file_matching_the_requested_quant(self, listing):
        listing(_tree(("M-Q4_K_M.gguf", 100), ("M-Q8_0.gguf", 200)))
        result = mp.remote_model_bytes("org/repo:Q8_0")
        assert result.total_bytes == 200 and result.primary_path == "M-Q8_0.gguf"

    def test_matches_the_quant_case_insensitively(self, listing):
        listing(_tree(("M-q4_k_m.gguf", 100)))
        assert mp.remote_model_bytes("org/repo:Q4_K_M").total_bytes == 100

    def test_a_quant_prefix_does_not_match_a_longer_quant(self, listing):
        """llama.cpp anchors the match as `tag[.-]`, so asking for Q4 must not silently resolve to
        Q4_K_M -- a different file, a different size, and not what would be downloaded."""
        listing(_tree(("M-Q4_K_M.gguf", 100)))
        assert mp.remote_model_bytes("org/repo:Q4") is None

    def test_sums_every_shard_of_a_split_model(self, listing):
        """All shards have to be resident to serve a token, so the first one's size is not the
        answer -- reporting it would understate a 70GB model as 35GB."""
        listing(_tree(
            ("M-Q8_0-00001-of-00003.gguf", 10),
            ("M-Q8_0-00002-of-00003.gguf", 20),
            ("M-Q8_0-00003-of-00003.gguf", 30),
        ))
        result = mp.remote_model_bytes("org/repo:Q8_0")
        assert result.total_bytes == 60
        assert len(result.paths) == 3

    def test_resolves_a_split_model_listed_out_of_order(self, listing):
        """The API returns whatever order it likes; the head shard is identified by its index, not
        by being first in the list."""
        listing(_tree(
            ("M-Q8_0-00002-of-00002.gguf", 20),
            ("M-Q8_0-00001-of-00002.gguf", 10),
        ))
        result = mp.remote_model_bytes("org/repo:Q8_0")
        assert result.total_bytes == 30
        assert result.primary_path.endswith("00001-of-00002.gguf")

    def test_does_not_count_a_sidecar_toward_the_model(self, listing):
        listing(_tree(("M-Q4_K_M.gguf", 100), ("mmproj-Q4_K_M.gguf", 7)))
        assert mp.remote_model_bytes("org/repo:Q4_K_M").total_bytes == 100

    def test_falls_back_to_llamacpps_default_quants_when_none_is_named(self, listing):
        listing(_tree(("M-Q8_0.gguf", 200), ("M-Q4_K_M.gguf", 100)))
        # Q4_K_M is first in llama.cpp's default order, regardless of listing order.
        assert mp.remote_model_bytes("org/repo").primary_path == "M-Q4_K_M.gguf"

    def test_no_verdict_when_the_quant_is_absent(self, listing):
        listing(_tree(("M-Q4_K_M.gguf", 100)))
        assert mp.remote_model_bytes("org/repo:Q2_K") is None

    def test_no_verdict_when_the_repo_has_no_gguf_at_all(self, listing):
        listing(_tree(("README.md", 10)))
        assert mp.remote_model_bytes("org/repo:Q4_K_M") is None

    def test_no_verdict_when_the_listing_cannot_be_read(self, listing):
        listing(None)
        assert mp.remote_model_bytes("org/repo:Q4_K_M") is None

    def test_no_verdict_when_sizes_are_missing(self, listing):
        """A listing with no size field is not evidence of a zero-byte model."""
        listing([{"type": "file", "path": "M-Q4_K_M.gguf"}])
        assert mp.remote_model_bytes("org/repo:Q4_K_M") is None

    def test_no_verdict_for_a_malformed_model_id(self):
        assert mp.remote_model_bytes("not-a-repo") is None


class TestFetchTreeFailureModes:
    """Every one of these has to mean "could not tell", never "does not fit" -- an offline laptop
    must not start refusing models."""

    def _urlopen_raising(self, exc):
        def _raise(*a, **kw):
            raise exc
        return _raise

    @pytest.mark.parametrize("exc", [
        urllib.error.URLError("offline"),
        OSError("connection reset"),
        TimeoutError("timed out"),
    ])
    def test_network_failures_yield_no_listing(self, monkeypatch, exc):
        monkeypatch.setattr(mp.urllib.request, "urlopen", self._urlopen_raising(exc))
        assert mp._fetch_tree("org/repo", timeout=1) is None

    def test_unparseable_json_yields_no_listing(self, monkeypatch):
        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self, *a): return b"not json"
        monkeypatch.setattr(mp.urllib.request, "urlopen", lambda *a, **kw: _Resp())
        assert mp._fetch_tree("org/repo", timeout=1) is None

    def test_a_non_list_payload_yields_no_listing(self, monkeypatch):
        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self, *a): return json.dumps({"error": "gated repo"}).encode()
        monkeypatch.setattr(mp.urllib.request, "urlopen", lambda *a, **kw: _Resp())
        assert mp._fetch_tree("org/repo", timeout=1) is None


class TestCheckFitsInMemory:
    def test_weights_larger_than_the_whole_machine_do_not_fit(self, listing):
        listing(_tree(("M-Q4_K_M.gguf", 8 * 1024**3)))
        verdict = mp.check_fits_in_memory("org/repo:Q4_K_M", total_memory_gb=4.0)
        assert not verdict.fits
        assert "do not fit" in verdict.detail

    def test_weights_that_merely_exceed_the_sizing_budget_still_fit(self, listing):
        """Deliberately NOT the 80% budget model_sizing uses. That budget answers "will this run
        well" and overshooting it means size conservatively; this answers "can this exist", and
        refusing a model that fits in RAM with the OS squeezed would be a false positive."""
        listing(_tree(("M-Q4_K_M.gguf", int(3.5 * 1024**3))))
        verdict = mp.check_fits_in_memory("org/repo:Q4_K_M", total_memory_gb=4.0)
        assert verdict.fits

    def test_no_verdict_when_memory_cannot_be_measured(self, listing):
        listing(_tree(("M-Q4_K_M.gguf", 8 * 1024**3)))
        assert mp.check_fits_in_memory("org/repo:Q4_K_M", total_memory_gb=None) is None

    def test_no_verdict_when_the_size_cannot_be_read(self, listing):
        listing(None)
        assert mp.check_fits_in_memory("org/repo:Q4_K_M", total_memory_gb=4.0) is None

    def test_the_detail_names_both_numbers_the_user_needs(self, listing):
        listing(_tree(("M-Q4_K_M.gguf", 8 * 1024**3)))
        detail = mp.check_fits_in_memory("org/repo:Q4_K_M", total_memory_gb=4.0).detail
        assert "8.0GB" in detail and "4.0GB" in detail

    def test_a_split_model_is_judged_on_its_total(self, listing):
        """The failure this guards: judging a 3-shard model on one shard and calling it a fit."""
        listing(_tree(
            ("M-Q4_K_M-00001-of-00002.gguf", 3 * 1024**3),
            ("M-Q4_K_M-00002-of-00002.gguf", 3 * 1024**3),
        ))
        assert not mp.check_fits_in_memory("org/repo:Q4_K_M", total_memory_gb=4.0).fits
