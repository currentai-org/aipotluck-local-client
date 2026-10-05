"""One-shot Hugging Face model pre-fetch, used by `aipotluck-local-client pull`.

Neither the installer nor the service ever downloads model weights on their own -- the installer
(install.py) only fetches the llama.cpp *binaries*, and llama-server itself lazily downloads
whatever `-hf`/`--model` it's configured with the first time it actually starts (see
service/runner.py's build_llama_server_args). This module exists so a user can trigger that
download ahead of time -- before pairing, or to switch models without waiting through a cold
start -- without reinventing Hugging Face's GGUF-resolution logic (matching a `repo:quant`
shorthand to the right file, split-GGUF handling, etc.). It reuses llama-server's own `-hf`
downloader for that, and only orchestrates around it: spawn on a private ephemeral port, poll
`/health` until the model has actually finished downloading AND loading, then terminate -- the
same "spawn, poll, terminate" shape service/llama_supervisor.py uses for the long-running
service, just run to completion once instead of supervised forever.
"""

from __future__ import annotations

import http.client
import logging
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger("aipotluck.installer.model_pull")

# A multi-GB quant over a slow connection can genuinely take a while -- this is a ceiling against
# a hung download, not a realistic expectation of how long every pull takes.
DEFAULT_TIMEOUT_SECONDS = 1800
HEALTH_POLL_INTERVAL_SECONDS = 2
LIST_TIMEOUT_SECONDS = 30

_CACHE_LIST_LINE_RE = re.compile(r"^\s*\d+\.\s+(\S.*\S|\S)\s*$")


class ModelPullError(RuntimeError):
    pass


def _free_port(host: str) -> int:
    """Ask the OS for an unused port, the same trick tests/test_runner.py's own `port=0` relies
    on -- but llama-server takes a literal port number on its command line, so we have to resolve
    one ourselves rather than letting the OS pick at bind time the way our own http.server does."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _check_health(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, http.client.HTTPException, OSError):
        return False


def pull_model(
    server_binary: Path,
    hf_target: str,
    *,
    host: str = "127.0.0.1",
    ctx_size: int | None = None,
    gpu_layers: str | int | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Downloads (or confirms already-cached) `hf_target` -- a Hugging Face `repo` or
    `repo:quant`, passed straight through to llama-server's own `-hf` flag -- blocking until
    llama-server reports healthy (downloaded AND successfully loaded) or `timeout` elapses.

    llama-server's own download progress is left going straight to this process's stdout/stderr
    (not captured) -- deliberately: a multi-GB download with no visible progress reads as a hang,
    and capturing it into a pipe risks deadlocking the child if it ever outpaces a pipe buffer
    nobody is draining. Raises ModelPullError on any failure; never returns partial success.
    """
    if not server_binary.exists():
        raise ModelPullError(f"llama-server binary not found at {server_binary}")

    port = _free_port(host)
    cmd = [str(server_binary), "-hf", hf_target, "--host", host, "--port", str(port)]
    if ctx_size:
        cmd += ["--ctx-size", str(ctx_size)]
    if gpu_layers is not None:
        cmd += ["--gpu-layers", str(gpu_layers)]

    log.info("Pulling %s via %s (this can take a while for a large quant)", hf_target, server_binary)
    proc = subprocess.Popen(cmd)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            exit_code = proc.poll()
            if exit_code is not None:
                raise ModelPullError(
                    f"llama-server exited (code={exit_code}) before becoming healthy while pulling "
                    f"{hf_target!r} -- see the output above for the real reason (a bad repo/quant "
                    "string is the most common one)."
                )
            if _check_health(host, port):
                log.info("%s downloaded and loads successfully", hf_target)
                return
            time.sleep(HEALTH_POLL_INTERVAL_SECONDS)
        raise ModelPullError(
            f"Timed out after {timeout:.0f}s waiting for {hf_target!r} to finish downloading/loading."
        )
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def list_cached_models(server_binary: Path) -> list[str]:
    """Returns every `repo:quant` target already downloaded locally, by asking llama-server's own
    `--cache-list` flag (also spelled `-cl`) -- it already reads the exact Hugging-Hub-compatible
    cache directory its own `-hf` downloader writes into (`$LLAMA_CACHE` / `$HF_HUB_CACHE` /
    `$HUGGINGFACE_HUB_CACHE` / `$HF_HOME/hub` / `$XDG_CACHE_HOME/huggingface/hub` /
    `~/.cache/huggingface/hub`, in that order -- see vendor/llama.cpp/common/hf-cache.cpp's
    get_cache_directory()), and already does its own filtering of multi-part/mmproj/draft-model
    files down to one entry per real model. Reusing it here keeps this in lockstep with whatever
    that resolution logic does, rather than re-implementing HF-cache-layout parsing a second time
    (the same reasoning pull_model above gives for reusing `-hf` instead of a custom downloader).

    `--cache-list` exits immediately after printing (no server ever starts), so this is a plain
    blocking subprocess call, not the spawn/poll/terminate dance `pull_model` needs.
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
