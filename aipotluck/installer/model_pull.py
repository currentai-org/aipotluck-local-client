"""One-shot Hugging Face model pre-fetch, used by `aipotluck-local-client pull`.

Neither the installer nor the service ever downloads model weights on their own -- the installer
(install.py) only fetches the llama.cpp *binaries*, and llama-server itself lazily downloads
whatever `-hf`/`--model` it's configured with the first time it actually starts (see
service/runner.py's build_llama_server_args). This module exists so a user can trigger that
download ahead of time -- before pairing, or to switch models without waiting through a cold
start -- without reinventing Hugging Face's GGUF-resolution logic (matching a `repo:quant`
shorthand to the right file, split-GGUF handling, etc.). The download itself goes through
llama-server's own router API on a scratch router (see cache_router), which is what reports
progress in bytes.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO

from aipotluck.installer import cache_router

log = logging.getLogger("aipotluck.installer.model_pull")

# A multi-GB quant over a slow connection can genuinely take a while -- this is a ceiling against
# a hung download, not a realistic expectation of how long every pull takes.
DEFAULT_TIMEOUT_SECONDS = 1800
LIST_TIMEOUT_SECONDS = 30

_CACHE_LIST_LINE_RE = re.compile(r"^\s*\d+\.\s+(\S.*\S|\S)\s*$")


class ModelPullError(RuntimeError):
    pass


def pull_model(
    server_binary: Path,
    hf_target: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    on_progress: cache_router.DownloadProgress | None = None,
) -> None:
    """Downloads `hf_target` -- a Hugging Face `repo` or `repo:quant` -- into the local cache, or
    returns at once if it is already there. `on_progress(received_bytes, total_bytes_or_None)` is
    called as bytes arrive. Raises ModelPullError on any failure; never returns partial success.

    It no longer loads the model to prove it works, as it did when the download ran through a
    plain `llama-server -hf`: sizing loads it straight afterwards anyway, and reports a model that
    cannot load far more precisely than a failed health check could."""
    log.info("Pulling %s (this can take a while for a large quant)", hf_target)
    try:
        cache_router.download(server_binary, hf_target, timeout=timeout, on_progress=on_progress)
    except cache_router.CacheRouterError as exc:
        raise ModelPullError(f"Could not download {hf_target!r}: {exc}") from exc


def terminal_progress(stream: TextIO | None = None, *, min_interval: float = 0.5) -> cache_router.DownloadProgress:
    """A progress callback for a person watching a terminal: one line redrawn in place on a TTY,
    and an occasional plain line otherwise (a log file or a pipe gets no carriage returns)."""
    out = stream or sys.stderr
    interactive = out.isatty()
    state = {"last": 0.0, "last_decile": -1}

    def report(received: int, total: int | None) -> None:
        now = time.monotonic()
        done = total is not None and received >= total
        if interactive:
            if not done and now - state["last"] < min_interval:
                return
            state["last"] = now
            if total:
                line = f"  downloading: {received / total:6.1%}  ({received / 1e6:,.1f} / {total / 1e6:,.1f} MB)"
            else:
                line = f"  downloading: {received / 1e6:,.1f} MB"
            out.write("\r" + line + ("\n" if done else ""))
            out.flush()
            return
        decile = int(received * 10 / total) if total else -1
        if decile > state["last_decile"]:
            state["last_decile"] = decile
            log.info("Downloaded %d%% (%.1f / %.1f MB)", decile * 10, received / 1e6, total / 1e6)

    return report


def list_cached_models(server_binary: Path) -> list[str]:
    """Returns every `repo:quant` target already downloaded locally, by asking llama-server's own
    `--cache-list` flag (also spelled `-cl`) -- it already reads the exact Hugging-Hub-compatible
    cache directory its own `-hf` downloader writes into (`$LLAMA_CACHE` / `$HF_HUB_CACHE` /
    `$HUGGINGFACE_HUB_CACHE` / `$HF_HOME/hub` / `$XDG_CACHE_HOME/huggingface/hub` /
    `~/.cache/huggingface/hub`, in that order -- see vendor/llama.cpp/common/hf-cache.cpp's
    get_cache_directory()), and already does its own filtering of multi-part/mmproj/draft-model
    files down to one entry per real model. Reusing it here keeps this in lockstep with whatever
    that resolution logic does, rather than re-implementing HF-cache-layout parsing a second time
    (the same reasoning pull_model above gives for reusing llama.cpp's own downloader).

    `--cache-list` exits immediately after printing (no server ever starts), so this is a plain
    blocking subprocess call.
    """
    if not server_binary.exists():
        raise ModelPullError(f"llama-server binary not found at {server_binary}")

    try:
        result = subprocess.run(
            [str(server_binary), "--cache-list"],
            # errors="replace" -- a cached model whose name or metadata decodes badly must not
            # make the whole cache unlistable. See model_sizing.probe_model_profile.
            capture_output=True, text=True, errors="replace", timeout=LIST_TIMEOUT_SECONDS,
        )
    except OSError as exc:
        raise ModelPullError(f"Could not run {server_binary} --cache-list: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ModelPullError(f"{server_binary} --cache-list did not exit within {LIST_TIMEOUT_SECONDS}s") from exc

    if result.returncode != 0:
        raise ModelPullError(
            f"{server_binary} --cache-list exited {result.returncode}:\n{result.stderr.strip()}"
        )

    models = []
    for line in result.stdout.splitlines():
        match = _CACHE_LIST_LINE_RE.match(line)
        if match:
            models.append(match.group(1))
    return models
