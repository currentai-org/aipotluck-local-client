"""aipotluck.installer.llama_errors -- reading llama.cpp's output when it has crashed.

The excerpt logic exists because of a live failure: a 24B model's vision projector failed to
allocate on a Jetson, llama.cpp printed the allocation error and then dumped ~30 lines of gdb
frames, and keeping the last 2,000 characters kept only the frames. The evidence was in the output
and was truncated away before anything could classify it.
"""

from __future__ import annotations

import pytest

from aipotluck.installer import llama_errors as le

# The real shape, reduced: the error, then the backtrace that buried it.
_REAL_CRASH = """\
0.20.455.690 D set_abort_callback: call
NvMapMemAllocInternalTagged: 1075072515 error 12
NvMapMemHandleAlloc: error 0
/home/ubuntu/src/vendor/llama.cpp/ggml/src/ggml-backend.cpp:188: GGML_ASSERT(buffer) failed
0.21.177.566 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 838.51 MiB on device 0: cudaMalloc failed: out of memory
0.21.177.582 E alloc_tensor_range: failed to allocate CUDA0 buffer of size 879243264
[New LWP 1075574]
[New LWP 1075575]
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/aarch64-linux-gnu/libthread_db.so.1".
0x0000ffffa24b9940 in __GI___wait4 (pid=<optimized out>, stat_loc=0x0) at ../sysdeps/unix/sysv/linux/wait4.c:30
30	../sysdeps/unix/sysv/linux/wait4.c: No such file or directory.
#0  0x0000ffffa24b9940 in __GI___wait4 (pid=<optimized out>) at ../sysdeps/unix/sysv/linux/wait4.c:30
#1  0x0000ffffa19c8e74 in ggml_print_backtrace () from /x/libggml-base.so.0
#2  0x0000ffffa19c9008 in ggml_abort () from /x/libggml-base.so.0
#3  0x0000ffffa1f6a1c8 in clip_model_loader::load_tensors(clip_ctx&) () from /x/libmtmd.so.0
#4  0x0000ffffa1f353c0 in clip_init(char const*, clip_context_params) () from /x/libmtmd.so.0
#5  0x0000ffffa1ea8368 in mtmd_context::mtmd_context(char const*) () from /x/libmtmd.so.0
#6  0x0000ffffa29be72c in server_context_impl::load_model(common_params&) () from /x/libllama-server-impl.so
#7  0x0000ffffa2477400 in __libc_start_call_main (main=0x1) at ../sysdeps/nptl/libc_start_call_main.h:58
#8  0x0000aaaac9f810f0 in _start ()
"""
# The real dump ran to roughly thirty frames and several dozen thread announcements, which is what
# pushed the error out of a 2,000-character tail. The fixture is padded to that length deliberately:
# at the real limit, a shorter dump would not reproduce the bug and the test below would pass for
# the wrong reason.
_REAL_CRASH += "".join(
    f"[New LWP {1075600 + i}]\n#{9 + i}  0x0000ffffa19c8e74 in frame_{i} () from /x/libggml-base.so.0\n"
    for i in range(30)
) + "[Inferior 1 (process 1072835) detached]\n"


class TestDiagnosticExcerpt:
    def test_the_error_survives_a_backtrace_that_would_have_buried_it(self):
        """The whole point. A blind tail of this output is pure gdb."""
        excerpt = le.diagnostic_excerpt(_REAL_CRASH, limit=400)
        assert "cudaMalloc failed: out of memory" in excerpt

    def test_a_blind_tail_would_have_missed_it(self):
        """Pins WHY the excerpt is needed rather than just that it works -- if llama.cpp ever stops
        printing backtraces this test is what says the complexity can go."""
        assert "cudaMalloc failed" not in _REAL_CRASH[-400:]

    def test_crash_dump_lines_are_dropped(self):
        excerpt = le.diagnostic_excerpt(_REAL_CRASH)
        assert "#3  0x" not in excerpt
        assert "[New LWP" not in excerpt
        assert "Thread debugging" not in excerpt

    def test_it_stays_within_the_limit(self):
        assert len(le.diagnostic_excerpt(_REAL_CRASH, limit=200)) <= 200

    def test_output_that_is_only_a_dump_still_returns_something(self):
        """An unhelpful excerpt beats an empty one -- a caller printing nothing at all would be
        worse than printing frames."""
        only_noise = "\n".join(f"#{i}  0x0000 in f () from /x/lib.so" for i in range(20))
        assert le.diagnostic_excerpt(only_noise).strip()

    def test_empty_output_is_empty(self):
        assert le.diagnostic_excerpt("") == ""

    def test_ordinary_output_is_left_alone(self):
        text = "load_model: loading model\nprint_info: n_ctx_train = 4096"
        assert le.diagnostic_excerpt(text) == text


class TestClassifyFailure:
    @pytest.mark.parametrize("line", [
        "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 512.00 MiB on device 0: cudaMalloc failed: out of memory",
        "llama_kv_cache: failed to allocate buffer for kv cache",
        "llama_model_load: unable to allocate CUDA0 buffer",
        "alloc_tensor_range: failed to allocate CUDA0 buffer of size 879243264",
        "terminate called after throwing an instance of 'std::bad_alloc'",
        "NvMapMemAllocInternalTagged: error 12",
    ])
    def test_real_allocation_failures_are_recognised(self, line):
        assert le.classify_failure(line, 1) == le.FAILURE_OOM

    def test_the_real_jetson_crash_classifies_as_an_oom_once_excerpted(self):
        """End to end on the captured output: excerpt, then classify."""
        excerpt = le.diagnostic_excerpt(_REAL_CRASH)
        assert le.classify_failure(excerpt, -6) == le.FAILURE_OOM

    def test_the_same_crash_is_misread_without_the_excerpt(self):
        """Documents the bug this module fixed, so a regression is unmistakable."""
        assert le.classify_failure(_REAL_CRASH[-2000:], -6) == le.FAILURE_OTHER

    def test_a_signal_kill_is_an_oom_even_with_nothing_logged(self):
        assert le.classify_failure("", -9) == le.FAILURE_OOM
        assert le.classify_failure("", 137) == le.FAILURE_OOM

    def test_an_unrelated_failure_is_not_an_oom(self):
        """The consequential direction: a corrupt file misread as an OOM would be 'fixed' by
        shrinking a context forever."""
        assert le.classify_failure(
            "llama_model_load: error loading model: invalid magic characters", 1
        ) == le.FAILURE_OTHER
