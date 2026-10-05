"""aipotluck.installer.model_sizing -- the CUR-1965 follow-up that computes per-model
ctx_size/parallel/cache_type_k/cache_type_v presets for llama-server's router mode.

Same "deliberately not mocked" posture as test_model_pull.py: probe_model_profile's subprocess
spawn, health polling, and log parsing are exercised against a real fake `llama-server` script,
not a mock that just returns what the test expects to see.
"""

from __future__ import annotations

import socket
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

from aipotluck.installer import model_presets, model_sizing
from aipotluck.installer.model_sizing import (
    ModelProfile,
    ModelSizingError,
    _kv_cache_bytes_per_token,
    _parse_int,
    _parse_int_or_list_max,
    _parse_model_size_bytes,
    compute_sizing,
    ensure_preset,
    probe_model_profile,
)

FAKE_SERVER_SCRIPT = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import argparse
    import sys
    import time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    sys.stdout.reconfigure(line_buffering=True)  # piped stdout is block-buffered by default --
    # without this, print_info lines sit in the child's buffer and never reach the parent before
    # SIGTERM kills it, the same way a real C binary's unbuffered stderr wouldn't.

    parser = argparse.ArgumentParser()
    parser.add_argument("-hf")
    parser.add_argument("--model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--ctx-size")
    parser.add_argument("--parallel")
    parser.add_argument("--gpu-layers")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail", action="store_true")
    parser.add_argument("--hang", action="store_true")
    parser.add_argument("--omit-hparams", action="store_true")
    parser.add_argument("--truncated-utf8", action="store_true")
    parser.add_argument("--n-ctx-train", default="131072")
    parser.add_argument("--n-layer", default="28")
    parser.add_argument("--n-embd-head-k", default="128")
    parser.add_argument("--n-embd-k-gqa", default="1024")
    parser.add_argument("--n-embd-v-gqa", default="1024")
    parser.add_argument("--file-size", default="1.92 GiB")
    args = parser.parse_args()

    if args.fail:
        print("error: could not resolve repo/quant", file=sys.stderr)
        sys.exit(1)

    if args.hang:
        time.sleep(600)
        sys.exit(0)

    if args.truncated_utf8:
        # The exact shape captured from a real Jetson run against LiquidAI/LFM2.5-8B-A1B-GGUF:
        # llama.cpp cuts its kv-preview at a fixed width without honouring UTF-8 boundaries, so a
        # two-byte `\u010a` (U+010A, b"\\xc4\\x8a" -- byte-level BPE's newline) loses its
        # trailing byte and the line ends on a lead byte. Emitted BEFORE the hparams on purpose:
        # under strict decoding the reader dies here and everything below never arrives, which is
        # why the real failure surfaced as a missing n_ctx_train.
        sys.stdout.flush()
        sys.stdout.buffer.write(
            b"llama_model_loader: - kv  30: tokenizer.ggml.tokens arr[str,128000] = "
            + b'["' + b"\\xc4\\x8a" * 5 + b'", "' + b"\\xc4\\x8a" * 5 + b"\\xc4...\\n"
        )
        sys.stdout.buffer.flush()

    if not args.omit_hparams:
        print("print_info: n_ctx_train           = " + args.n_ctx_train)
        print("print_info: n_layer               = " + args.n_layer)
        print("print_info: n_layer_all           = " + args.n_layer)
        print("print_info: n_embd_head_k         = " + args.n_embd_head_k)
        print("print_info: n_embd_k_gqa          = " + args.n_embd_k_gqa)
        print("print_info: n_embd_v_gqa          = " + args.n_embd_v_gqa)
        print("load_tensors: file size   = " + args.file_size + " (4.50 BPW) ")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *a):
            pass

    HTTPServer((args.host, args.port), Handler).serve_forever()
    """
)


@pytest.fixture
def fake_server_binary(tmp_path: Path) -> Path:
    script = tmp_path / "fake-llama-server"
    script.write_text(FAKE_SERVER_SCRIPT, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


def _port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


class TestParseHelpers:
    def test_parse_int_reads_a_plain_value(self):
        assert _parse_int("print_info: n_ctx_train           = 131072\n", "n_ctx_train") == 131072

    def test_parse_int_does_not_match_a_longer_key_sharing_the_same_prefix(self):
        # "n_layer_all" must never satisfy a lookup for "n_layer" -- the regex requires only
        # whitespace between the key and "=", and "_all" isn't whitespace.
        text = "print_info: n_layer_all           = 999\nprint_info: n_layer               = 28\n"
        assert _parse_int(text, "n_layer") == 28

    def test_parse_int_raises_when_key_absent(self):
        with pytest.raises(ModelSizingError, match="n_ctx_train"):
            _parse_int("nothing useful here", "n_ctx_train")

    def test_parse_int_or_list_max_reads_a_plain_value(self):
        assert _parse_int_or_list_max("n_embd_k_gqa          = 1024\n", "n_embd_k_gqa") == 1024

    def test_parse_int_or_list_max_takes_the_max_of_a_bracketed_list(self):
        # Heterogeneous (e.g. MoE) architectures print per-layer values as "[a, b, c]" --
        # taking the max is the safe-to-over-budget direction for a memory sizing decision.
        assert _parse_int_or_list_max("n_embd_k_gqa          = [512, 1024, 768]\n", "n_embd_k_gqa") == 1024

    def test_parse_int_or_list_max_raises_on_an_empty_list(self):
        with pytest.raises(ModelSizingError, match="empty list"):
            _parse_int_or_list_max("n_embd_k_gqa          = []\n", "n_embd_k_gqa")

    def test_parse_model_size_bytes_handles_gib(self):
        text = "load_tensors: file size   = 1.92 GiB (4.50 BPW) \n"
        assert _parse_model_size_bytes(text) == int(1.92 * 1024**3)

    def test_parse_model_size_bytes_handles_mib(self):
        text = "load_tensors: file size   = 350.00 MiB (4.50 BPW) \n"
        assert _parse_model_size_bytes(text) == int(350.0 * 1024**2)

    def test_parse_model_size_bytes_raises_when_absent(self):
        with pytest.raises(ModelSizingError, match="file size"):
            _parse_model_size_bytes("nothing useful here")


class TestProbeModelProfile:
    def test_passes_verbose_so_print_info_lines_are_never_gated_out(self, fake_server_binary, monkeypatch):
        # CUR-1965, found live against a real device: at least one real llama-server build gates
        # its print_info: hparam lines behind a verbosity threshold not met at default settings --
        # the vendored source alone didn't predict this. --verbose is the one setting guaranteed
        # to surface them regardless of a given build's default threshold.
        monkeypatch.setattr(model_sizing, "_total_memory_gb", lambda: 3.5)
        real_popen = subprocess.Popen
        seen_cmds = []

        def _capture_popen(cmd, *args, **kwargs):
            seen_cmds.append(cmd)
            return real_popen(cmd, *args, **kwargs)

        monkeypatch.setattr(model_sizing.subprocess, "Popen", _capture_popen)
        probe_model_profile(fake_server_binary, model_hf="org/model:Q4_K_M")
        assert seen_cmds and "--verbose" in seen_cmds[0]

    def test_returns_a_profile_parsed_from_the_real_server_output(self, fake_server_binary, monkeypatch):
        monkeypatch.setattr(model_sizing, "_total_memory_gb", lambda: 3.5)
        profile = probe_model_profile(fake_server_binary, model_hf="org/model:Q4_K_M")
        assert profile == ModelProfile(
            n_ctx_train=131072, n_layer=28, n_embd_head_k=128, n_embd_k_gqa=1024, n_embd_v_gqa=1024,
            model_size_bytes=int(1.92 * 1024**3), total_memory_gb=3.5,
        )

    def test_terminates_the_probe_process_rather_than_leaving_it_running(self, fake_server_binary, monkeypatch):
        monkeypatch.setattr(model_sizing, "_total_memory_gb", lambda: 3.5)
        port_holder: list[int] = []
        real_free_port = model_sizing._free_port

        def _capture_port(host):
            port = real_free_port(host)
            port_holder.append(port)
            return port

        monkeypatch.setattr(model_sizing, "_free_port", _capture_port)
        probe_model_profile(fake_server_binary, model_hf="org/model:Q4_K_M")
        assert port_holder, "the fake port-allocator was never called"
        assert not _port_is_open("127.0.0.1", port_holder[0]), "probe process was left running after profiling"

    def test_raises_when_the_server_exits_before_becoming_healthy(self, fake_server_binary, monkeypatch):
        # A bad repo/quant string is the realistic real-world trigger for this.
        real_popen = subprocess.Popen

        def _fake_popen(cmd, *args, **kwargs):
            return real_popen([*cmd, "--fail"], *args, **kwargs)

        monkeypatch.setattr(model_sizing.subprocess, "Popen", _fake_popen)
        with pytest.raises(ModelSizingError, match="exited"):
            probe_model_profile(fake_server_binary, model_hf="org/does-not-exist:Q4_K_M")

    def test_survives_a_half_written_utf8_character_in_the_server_log(
        self, fake_server_binary, monkeypatch
    ):
        """llama.cpp truncates its kv preview without honouring UTF-8 boundaries, so a byte-level
        BPE tokenizer can leave a lead byte with no continuation. Confirmed live on a Jetson with
        LiquidAI/LFM2.5-8B-A1B-GGUF, where strict decoding killed the reader thread and the pull
        died. The contract is that undecodable bytes cost nothing: the probe still returns the
        hparams printed AFTER them."""
        monkeypatch.setattr(model_sizing, "_total_memory_gb", lambda: 3.5)
        real_popen = subprocess.Popen

        def _fake_popen(cmd, *args, **kwargs):
            return real_popen([*cmd, "--truncated-utf8"], *args, **kwargs)

        monkeypatch.setattr(model_sizing.subprocess, "Popen", _fake_popen)
        profile = probe_model_profile(fake_server_binary, model_hf="LiquidAI/LFM2.5-8B-A1B:Q4_K_M")
        assert profile.n_ctx_train == 131072
        assert profile.n_layer == 28

    def test_blames_the_reader_when_output_was_cut_short(self, fake_server_binary, monkeypatch):
        """A truncated read and a genuinely absent hparam are indistinguishable at the parse step --
        both are just a key missing from the text. When the reader is why, the error has to say so,
        or the next person debugs the regex instead of the pipe."""
        monkeypatch.setattr(model_sizing, "_total_memory_gb", lambda: 3.5)
        real_drain = model_sizing._drain

        def _exploding_drain(proc, lines, failures):
            # Let nothing through, exactly as a dead reader thread would.
            failures.append(UnicodeDecodeError("utf-8", b"\xc4", 0, 1, "invalid continuation byte"))

        monkeypatch.setattr(model_sizing, "_drain", _exploding_drain)
        with pytest.raises(ModelSizingError, match="cut short by a read error"):
            probe_model_profile(fake_server_binary, model_hf="org/model:Q4_K_M")
        assert real_drain is not None  # the real one is restored by monkeypatch teardown

    def test_raises_on_timeout_rather_than_hanging_forever(self, fake_server_binary, monkeypatch):
        real_popen = subprocess.Popen

        def _fake_popen(cmd, *args, **kwargs):
            return real_popen([*cmd, "--hang"], *args, **kwargs)

        monkeypatch.setattr(model_sizing.subprocess, "Popen", _fake_popen)
        monkeypatch.setattr(model_sizing, "_HEALTH_POLL_INTERVAL_SECONDS", 0.1)
        with pytest.raises(ModelSizingError, match="Timed out"):
            probe_model_profile(fake_server_binary, model_hf="org/model:Q4_K_M", timeout=0.5)

    def test_raises_when_a_required_hparam_is_missing_from_the_output(self, tmp_path, monkeypatch):
        # A minimal, separate fake binary rather than reusing fake_server_binary with an extra
        # flag: probe_model_profile's own command builder has no passthrough for arbitrary flags
        # (deliberately -- it only knows the flags a real sizing probe needs), so the cleanest real
        # reproduction of "the server started but its log doesn't have what we need" is a fake
        # binary that never prints the hparam lines at all.
        script = tmp_path / "fake-llama-server-no-hparams"
        script.write_text(
            FAKE_SERVER_SCRIPT.replace('parser.add_argument("--omit-hparams", action="store_true")',
                                        'parser.add_argument("--omit-hparams", action="store_true", default=True)'),
            encoding="utf-8",
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        monkeypatch.setattr(model_sizing, "_total_memory_gb", lambda: 3.5)
        with pytest.raises(ModelSizingError, match="n_ctx_train"):
            probe_model_profile(script, model_hf="org/model:Q4_K_M")


class TestKvCacheBytesPerToken:
    def test_matches_the_real_measured_llama_3_2_3b_figure(self):
        # Llama 3.2 3B: 28 layers, 8 KV-heads (GQA), 128 head-dim, f16 -- confirmed against the
        # real device this whole feature was built for (CUR-1965): ~112 KiB/token.
        profile = ModelProfile(
            n_ctx_train=131072, n_layer=28, n_embd_head_k=128, n_embd_k_gqa=1024, n_embd_v_gqa=1024,
            model_size_bytes=0, total_memory_gb=None,
        )
        bytes_per_token = _kv_cache_bytes_per_token(profile, "f16", "f16")
        assert bytes_per_token == 28 * (1024 * 2 + 1024 * 2)
        assert round(bytes_per_token / 1024, 1) == 112.0

    def test_q8_0_is_roughly_half_of_f16(self):
        profile = ModelProfile(
            n_ctx_train=8192, n_layer=28, n_embd_head_k=128, n_embd_k_gqa=1024, n_embd_v_gqa=1024,
            model_size_bytes=0, total_memory_gb=None,
        )
        f16 = _kv_cache_bytes_per_token(profile, "f16", "f16")
        q8_0 = _kv_cache_bytes_per_token(profile, "q8_0", "q8_0")
        assert 0.5 < q8_0 / f16 < 0.6


def _profile(
    total_memory_gb: float | None, *, n_ctx_train=8192, n_embd_head_k=128, model_size_bytes=0
) -> ModelProfile:
    return ModelProfile(
        n_ctx_train=n_ctx_train, n_layer=28, n_embd_head_k=n_embd_head_k,
        n_embd_k_gqa=1024, n_embd_v_gqa=1024, model_size_bytes=model_size_bytes,
        total_memory_gb=total_memory_gb,
    )


class TestComputeSizing:
    def test_always_sizes_to_a_single_parallel_slot(self):
        result = compute_sizing(_profile(3.0))
        assert result.parallel == 1
        assert "parallel" in result.tuning

    def test_prefers_q8_0_when_it_reaches_the_full_trained_context(self):
        result = compute_sizing(_profile(3.0))
        assert result.cache_type_k == result.cache_type_v == "q8_0"
        assert result.ctx_size == 8192  # capped at n_ctx_train, not the much larger q8 max

    def test_degrades_to_q4_0_when_only_q4_0_reaches_the_full_trained_context(self):
        result = compute_sizing(_profile(0.45))
        assert result.cache_type_k == result.cache_type_v == "q4_0"
        assert result.ctx_size == 8192

    def test_sizes_to_the_largest_context_that_fits_when_neither_quant_reaches_full_context(self):
        result = compute_sizing(_profile(0.08))
        assert result.cache_type_k == result.cache_type_v == "q4_0"
        assert result.ctx_size < 8192
        assert result.ctx_size >= model_sizing._MIN_CTX_SIZE

    def test_falls_back_to_the_min_ctx_floor_rather_than_a_too_small_context(self):
        result = compute_sizing(_profile(0.001))
        assert result.ctx_size == model_sizing._MIN_CTX_SIZE


class TestSizingViability:
    """Whether this device can give the model a context big enough to hold one turn.

    Sizing reaches this verdict from the memory budget alone, before anything is downloaded into a
    benchmark, so it is the cheapest place to catch a model that has nowhere to put a conversation.
    Before this, such a model was floored to 512 tokens and returned as though it had been sized:
    the preset looked ordinary and the failure surfaced much later.
    """

    def test_a_comfortable_model_is_viable(self):
        result = compute_sizing(_profile(3.0))
        assert result.viable
        assert result.rejection_code is None and result.rejection is None

    def test_rejects_when_the_weights_leave_no_room_for_a_kv_cache(self):
        # 8GB of weights against 4GB of RAM: the budget is negative before context is considered.
        result = compute_sizing(_profile(4.0, model_size_bytes=8 * 1024**3))
        assert not result.viable
        assert result.rejection_code == model_sizing.REJECT_NO_ROOM_FOR_CONTEXT
        assert "exceed the budget" in result.rejection

    def test_rejects_when_room_exists_but_is_under_one_turn(self):
        """The subtler half: memory is left over, just not enough for a turn. This is the case the
        512-token floor used to hide."""
        result = compute_sizing(_profile(0.001))
        assert not result.viable
        assert result.rejection_code == model_sizing.REJECT_NO_ROOM_FOR_CONTEXT
        assert "one turn needs 2048" in result.rejection

    def test_still_returns_a_usable_sizing_alongside_the_rejection(self):
        """compute_sizing's never-raises contract is unchanged: callers that only want numbers
        keep getting them, and only the ones that ask about `viable` change behaviour."""
        result = compute_sizing(_profile(0.001))
        assert result.ctx_size == model_sizing._MIN_CTX_SIZE
        assert result.parallel == 1
        assert result.tuning["ctx_size"]

    def test_blames_the_model_not_the_device_when_the_trained_context_is_tiny(self):
        """A 1k-token model fails on any hardware, so the reason must not point at this device --
        otherwise a user goes looking for more RAM that would not have helped."""
        result = compute_sizing(_profile(64.0, n_ctx_train=1024))
        assert not result.viable
        assert result.rejection_code == model_sizing.REJECT_CONTEXT_TOO_SMALL
        assert "trained context" in result.rejection

    def test_is_viable_at_exactly_the_minimum_servable_context(self):
        """Pins the boundary: 2048 is servable, so the gate must not be off by one."""
        profile = _profile(64.0, n_ctx_train=model_sizing.MIN_SERVABLE_CTX_TOKENS)
        assert compute_sizing(profile).viable

    def test_decides_a_tiny_trained_context_even_when_memory_cannot_be_measured(self):
        """On a platform with no /proc/meminfo there is no budget to reason about, but a model
        trained too narrow is a fact about the model, so it is still decidable."""
        result = compute_sizing(_profile(None, n_ctx_train=1024))
        assert not result.viable
        assert result.rejection_code == model_sizing.REJECT_CONTEXT_TOO_SMALL

    def test_stays_viable_when_memory_cannot_be_measured_and_the_model_is_wide_enough(self):
        """Must not guess. With no memory reading, an adequately-trained model gets the benefit of
        the doubt rather than a rejection built on a number we do not have."""
        result = compute_sizing(_profile(None, n_ctx_train=8192))
        assert result.viable and result.rejection is None

    def test_the_models_own_weight_size_is_subtracted_from_the_budget_before_sizing_context(self):
        # Same total RAM (1.0GB) and same trained context (8192, the _profile default) -- but one
        # profile's model weighs 0.9GB, leaving next to nothing for KV cache. Chosen so that
        # WITHOUT the subtraction both would still comfortably reach the full trained context
        # (making a weaker "<=" assertion pass even if the subtraction were silently dropped) --
        # strict "<" only holds if the model's own weight genuinely came out of the budget.
        small_model = compute_sizing(_profile(1.0, model_size_bytes=0))
        big_model = compute_sizing(_profile(1.0, model_size_bytes=int(0.9 * 1024**3)))
        assert small_model.ctx_size == 8192
        assert big_model.ctx_size < small_model.ctx_size
        assert big_model.ctx_size == model_sizing._MIN_CTX_SIZE  # budget went negative -- floors

    def test_skips_kv_quantization_when_head_dim_does_not_divide_the_quant_block_size(self):
        # head_dim=80 -- 80 % 32 != 0, so llama.cpp's own startup validation would refuse a
        # quantized K cache for this architecture (confirmed against llama-context.cpp).
        result = compute_sizing(_profile(3.0, n_embd_head_k=80))
        assert result.cache_type_k is None
        assert result.cache_type_v is None
        assert "cache_type_k" in result.tuning

    def test_falls_back_to_trained_context_with_no_quantization_when_memory_cannot_be_measured(self):
        # Fail-open path: no memory reading (e.g. non-Linux today) must not silently pick an
        # unbounded or zero context -- it must fall back to the model's own trained size.
        result = compute_sizing(_profile(None, n_ctx_train=4096))
        assert result.ctx_size == 4096
        assert result.cache_type_k is None
        assert result.cache_type_v is None
        assert "could not measure" in result.tuning["ctx_size"]


class TestEnsurePreset:
    def test_writes_a_preset_and_returns_the_sizing_result(self, fake_server_binary, tmp_path, monkeypatch):
        monkeypatch.setattr(model_sizing, "_total_memory_gb", lambda: 3.0)
        presets_path = tmp_path / "presets.ini"
        result = ensure_preset(fake_server_binary, presets_path, "org/model:Q4_K_M", model_hf="org/model:Q4_K_M")
        assert result is not None
        assert model_presets.has_preset(presets_path, "org/model:Q4_K_M")
        text = presets_path.read_text(encoding="utf-8")
        assert f"ctx-size = {result.ctx_size}" in text
        assert model_presets.read_tuning(presets_path)["org/model:Q4_K_M"] == result.tuning

    def test_skips_recompute_when_a_preset_already_exists_and_not_forced(self, fake_server_binary, tmp_path, monkeypatch):
        monkeypatch.setattr(model_sizing, "_total_memory_gb", lambda: 3.0)
        presets_path = tmp_path / "presets.ini"
        model_presets.write_preset(presets_path, "org/model:Q4_K_M", {"ctx-size": "999"})

        probe_calls = []
        monkeypatch.setattr(model_sizing, "probe_model_profile", lambda *a, **kw: probe_calls.append(1))

        result = ensure_preset(fake_server_binary, presets_path, "org/model:Q4_K_M", model_hf="org/model:Q4_K_M")

        assert result is None
        assert probe_calls == []
        assert "ctx-size = 999" in presets_path.read_text(encoding="utf-8")  # untouched

    def test_force_recomputes_even_when_a_preset_already_exists(self, fake_server_binary, tmp_path, monkeypatch):
        monkeypatch.setattr(model_sizing, "_total_memory_gb", lambda: 3.0)
        presets_path = tmp_path / "presets.ini"
        model_presets.write_preset(presets_path, "org/model:Q4_K_M", {"ctx-size": "999"})

        result = ensure_preset(
            fake_server_binary, presets_path, "org/model:Q4_K_M", model_hf="org/model:Q4_K_M", force=True
        )

        assert result is not None
        assert "ctx-size = 999" not in presets_path.read_text(encoding="utf-8")

    def test_omits_cache_type_keys_when_the_architecture_cannot_use_them(self, tmp_path, monkeypatch):
        script = tmp_path / "fake-llama-server-odd-head-dim"
        script.write_text(
            FAKE_SERVER_SCRIPT.replace('default="128"', 'default="80"'),
            encoding="utf-8",
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        monkeypatch.setattr(model_sizing, "_total_memory_gb", lambda: 3.0)
        presets_path = tmp_path / "presets.ini"

        ensure_preset(script, presets_path, "org/model:Q4_K_M", model_hf="org/model:Q4_K_M")

        text = presets_path.read_text(encoding="utf-8")
        assert "cache-type-k" not in text
        assert "cache-type-v" not in text
