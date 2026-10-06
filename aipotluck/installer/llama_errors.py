"""Reading llama.cpp's output when something has gone wrong.

Two jobs, both shared between the installer and the service, which is why they live down here
rather than in either one: deciding whether a failure was an out-of-memory, and pulling the part of
the output that says so out of the part that does not.

The second job is not cosmetic. When llama.cpp aborts it prints the real error and THEN dumps a
backtrace, so the tail of its output -- the obvious thing to keep -- is the least informative part
of it. A 24B model failing to load on a Jetson produced roughly thirty lines of gdb frames after
the one line naming the allocation it could not make, and keeping the last 2,000 characters kept
only the frames. The evidence was there, in the output, and got truncated away before anything
could read it.
"""

from __future__ import annotations

import re

# Allocation-failure strings, taken from the vendored llama.cpp rather than invented: a lookup that
# misses means a real OOM is classified as "some other failure" and silently never recovered from.
#   ggml/src/ggml-cuda/ggml-cuda.cu:893   "allocating ... cudaMalloc failed: out of memory"
#   src/llama-kv-cache.cpp:288            "failed to allocate buffer for kv cache"
#   src/llama-model.cpp:1767,1783         "unable to allocate <buft> buffer"
#   src/llama-context.cpp:644,656,681     "failed to allocate compute pp/tg buffers"
#   src/llama-context.cpp:2463            "failed to allocate compute buffers"
#   src/llama-context.cpp:2132            "failed to allocate output buffer of size"
#   ggml/src/ggml-backend.cpp:2448        "failed to allocate buffer of size"
# The last three entries are not llama.cpp's: `std::bad_alloc` is what an uncaught host-allocation
# failure prints as it aborts, NvMapMemAllocInternalTagged is how a Jetson's unified memory reports
# exhaustion, and alloc_tensor_range is ggml's own wrapper around a backend buffer it could not get
# (seen live on an Orin NX loading a 24B model's vision projector).
OOM_LOG_SIGNATURES = (
    "cudamalloc failed: out of memory",
    "failed to allocate buffer for kv cache",
    "unable to allocate",
    "failed to allocate compute",
    "failed to allocate output buffer",
    "failed to allocate buffer of size",
    "failed to allocate graph",
    "alloc_tensor_range: failed to allocate",
    "std::bad_alloc",
    "cannot allocate memory",
    "nvmapmemallocinternaltagged",
    "out of memory",
)

# A child killed by the Linux OOM killer never gets to print anything, so the exit code is the only
# evidence there is. 137 is a shell's 128+SIGKILL; -9 is what Python's subprocess reports for the
# same thing, and a caller passes through whichever its own platform produced.
OOM_EXIT_CODES = (137, -9)

FAILURE_OOM = "oom"
FAILURE_OTHER = "other"

# Lines that are part of a crash dump rather than part of the diagnosis. Dropping these is what
# lets a bounded excerpt still contain the error: gdb frames, thread announcements and missing
# source-file notes are generated in bulk and crowd out everything else.
_NOISE_PATTERNS = (
    re.compile(r"^#\d+\s"),                       # gdb stack frames
    re.compile(r"^\[(New|Thread|Inferior)\b"),    # thread/inferior announcements
    re.compile(r"^0x[0-9a-fA-F]+\s+in\s"),        # the frame gdb stops at
    re.compile(r"^\d+\s+\.\./"),                  # source lines gdb cannot find
    re.compile(r"^(\.\./|/usr/src/).*No such file or directory"),
    re.compile(r"^Using host libthread_db"),
    re.compile(r"^\[Thread debugging using"),
    re.compile(r"^\s*$"),
)


def _is_noise(line: str) -> bool:
    return any(pattern.match(line) for pattern in _NOISE_PATTERNS)


def diagnostic_excerpt(output: str, *, limit: int = 2000) -> str:
    """The most useful `limit` characters of llama.cpp's output.

    Crash-dump lines are dropped first and the tail of what remains is kept, so the excerpt holds
    the error rather than the backtrace that buried it. If filtering leaves nothing -- a failure
    that produced only a dump -- the raw tail is returned, because an unhelpful excerpt is still
    better than an empty one."""
    if not output:
        return ""
    kept = [line for line in output.splitlines() if not _is_noise(line)]
    if not kept:
        return output[-limit:]
    return "\n".join(kept)[-limit:]


def classify_failure(log_tail: str, exit_code: int | None) -> str:
    """Was this process's death an out-of-memory, or something else?

    Log evidence wins over the exit code, because it is specific: llama.cpp names the allocation it
    could not make. The exit code is only consulted when there is nothing in the log to read, which
    is exactly the OOM-killer case -- SIGKILL gives the process no chance to explain itself."""
    haystack = (log_tail or "").lower()
    for signature in OOM_LOG_SIGNATURES:
        if signature in haystack:
            return FAILURE_OOM
    if exit_code in OOM_EXIT_CODES:
        return FAILURE_OOM
    return FAILURE_OTHER
