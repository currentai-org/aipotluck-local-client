"""A short-lived llama-server router with no presets, for operating on the model cache itself.

Removing a model has to go through llama.cpp's own cache logic (`common_download_remove` clears the
snapshot, its symlinks and the blobs nothing else references) rather than a reimplementation of the
HF cache layout here. The only way to reach that logic is a router's `DELETE /models`, and the
service's own long-running router cannot be relied on for it. Both failures below were reproduced
against the pinned b10989 binary, not inferred from the source:

  - It refuses any model named in `--models-preset` ("is not removable (not from cache)"), and
    every model this project sizes has a preset. Dropping the preset and asking it to reload does
    not help: its reload refreshes a model's preset but never the source it recorded at startup,
    so the refusal lasts until the process restarts.
  - It only learns about the cache when it reloads, so a model downloaded since then -- which is
    exactly the one a rejected pull needs to delete -- is "not found".

A router started fresh with no presets file has neither problem: it reads the cache on startup and
sees every model in it as cache-sourced. It starts in well under a second because it loads nothing.
It shares the main router's cache by inheriting the same environment ($LLAMA_CACHE, $HF_HOME, ...),
which is also why this works when the service is stopped or logged out.
"""

from __future__ import annotations

import contextlib
import json
import logging
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterator

from aipotluck.installer.model_pull import _check_health, _free_port

log = logging.getLogger("aipotluck.installer.cache_router")

_HOST = "127.0.0.1"
STARTUP_TIMEOUT_SECONDS = 30.0
DELETE_TIMEOUT_SECONDS = 60.0


class CacheRouterError(RuntimeError):
    pass


def error_message(exc: urllib.error.HTTPError) -> str:
    """The router's own explanation from an error response, which is far more useful to a person
    than the status code -- `{"error": {"message": ...}}` is llama-server's error shape."""
    try:
        body = exc.read().decode("utf-8", "replace")
    except OSError:
        return f"HTTP {exc.code}"
    try:
        message = json.loads(body).get("error", {}).get("message")
    except (json.JSONDecodeError, AttributeError):
        message = None
    return message or body.strip()[:300] or f"HTTP {exc.code}"


@contextlib.contextmanager
def scratch_router(server_binary: Path, *, startup_timeout: float = STARTUP_TIMEOUT_SECONDS) -> Iterator[str]:
    """Runs a preset-free router for the duration of the block, yielding its base URL."""
    if not server_binary.exists():
        raise CacheRouterError(f"llama-server binary not found at {server_binary}")

    port = _free_port(_HOST)
    cmd = [str(server_binary), "--host", _HOST, "--port", str(port)]
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise CacheRouterError(f"could not start {server_binary}: {exc}") from exc

    try:
        deadline = time.monotonic() + startup_timeout
        while not _check_health(_HOST, port):
            if proc.poll() is not None:
                raise CacheRouterError(
                    f"llama-server exited (code={proc.returncode}) before its router came up"
                )
            if time.monotonic() >= deadline:
                raise CacheRouterError(f"llama-server's router did not come up within {startup_timeout:.0f}s")
            time.sleep(0.1)
        yield f"http://{_HOST}:{port}"
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def delete(server_binary: Path, model_id: str) -> None:
    """Removes `model_id`'s files from the cache. Raises CacheRouterError with llama.cpp's own reason
    on failure.

    A copy of the model that another router is serving keeps running from the unlinked files on
    POSIX, and on Windows holds them open so they cannot be deleted -- unload it there first."""
    with scratch_router(server_binary) as base_url:
        url = f"{base_url}/models?model={urllib.parse.quote(model_id, safe='')}"
        request = urllib.request.Request(url, method="DELETE")
        try:
            with urllib.request.urlopen(request, timeout=DELETE_TIMEOUT_SECONDS):
                return
        except urllib.error.HTTPError as exc:
            raise CacheRouterError(error_message(exc)) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise CacheRouterError(f"lost contact with llama-server while removing {model_id}: {exc}") from exc
