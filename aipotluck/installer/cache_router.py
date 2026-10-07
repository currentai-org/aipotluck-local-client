"""A short-lived llama-server router with no presets, for operating on the model cache itself.

Downloading and removing models both happen here, through the router's own documented API
(`POST /models` and `DELETE /models`), so llama.cpp's own code resolves `repo:quant` to files and
lays them out in the cache -- nothing here reimplements that layout.

**Removing.** It has to go through llama.cpp's own cache logic (`common_download_remove` clears the
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

**Downloading.** The router reports a download's progress per file, in bytes, on its event stream
(`GET /models/sse`), which is what lets a caller show a real progress bar. Downloading through the
service's own router would leave it running the download inside the very process that serves turns,
and would not work while the service is stopped, so this uses a scratch router too -- which also
means the service's router learns about the model on its next reload rather than mid-download.

One thing llama.cpp gets wrong, confirmed against b10989: a repo that does not exist reports
`download_finished` all the same (Hugging Face answers 401, and the downloader logs it and carries
on). So "finished" is never taken as success on its own -- the model has to be in the cache
afterwards, and when it isn't, the router's own last error line says why.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import logging
import os
import queue
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

log = logging.getLogger("aipotluck.installer.cache_router")

_HOST = "127.0.0.1"
STARTUP_TIMEOUT_SECONDS = 30.0
DELETE_TIMEOUT_SECONDS = 60.0
# POST /models fetches the repo's metadata from Hugging Face before it answers.
REQUEST_TIMEOUT_SECONDS = 60.0

# (bytes received so far, total bytes or None when not yet known)
DownloadProgress = Callable[[int, "int | None"], None]


class CacheRouterError(RuntimeError):
    pass


def _free_port(host: str) -> int:
    """llama-server takes a literal port number on its command line, so one has to be resolved
    up front rather than letting the OS pick at bind time."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _check_health(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, http.client.HTTPException, OSError):
        return False


@dataclass
class ScratchRouter:
    base_url: str
    log_path: Path

    def last_error(self) -> str | None:
        """The last error line llama-server logged, which is usually the real reason for a failure
        its HTTP API reported as something vaguer -- or as success."""
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return None
        for line in reversed(lines):
            # llama.cpp's log format puts the level letter after the timestamp: "... E get_repo_..."
            parts = line.split(maxsplit=2)
            for i, part in enumerate(parts[:2]):
                if part == "E" and i + 1 < len(parts):
                    return " ".join(parts[i + 1:]).strip()
        return None


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
def scratch_router(server_binary: Path, *, startup_timeout: float = STARTUP_TIMEOUT_SECONDS) -> Iterator[ScratchRouter]:
    """Runs a preset-free router for the duration of the block. Its output goes to a temporary log
    file, removed afterwards, so a failure can quote it."""
    if not server_binary.exists():
        raise CacheRouterError(f"llama-server binary not found at {server_binary}")

    port = _free_port(_HOST)
    cmd = [str(server_binary), "--host", _HOST, "--port", str(port)]
    log_fd, log_name = tempfile.mkstemp(prefix="aipotluck-cache-router-", suffix=".log")
    log_path = Path(log_name)
    try:
        with os.fdopen(log_fd, "wb") as log_file:
            try:
                proc = subprocess.Popen(
                    cmd, stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT,
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
                        raise CacheRouterError(
                            f"llama-server's router did not come up within {startup_timeout:.0f}s"
                        )
                    time.sleep(0.1)
                yield ScratchRouter(f"http://{_HOST}:{port}", log_path)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
    finally:
        log_path.unlink(missing_ok=True)


def delete(server_binary: Path, model_id: str) -> None:
    """Removes `model_id`'s files from the cache. Raises CacheRouterError with llama.cpp's own reason
    on failure.

    A copy of the model that another router is serving keeps running from the unlinked files on
    POSIX, and on Windows holds them open so they cannot be deleted -- unload it there first."""
    with scratch_router(server_binary) as router:
        url = f"{router.base_url}/models?model={urllib.parse.quote(model_id, safe='')}"
        request = urllib.request.Request(url, method="DELETE")
        try:
            with urllib.request.urlopen(request, timeout=DELETE_TIMEOUT_SECONDS):
                return
        except urllib.error.HTTPError as exc:
            raise CacheRouterError(error_message(exc)) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise CacheRouterError(f"lost contact with llama-server while removing {model_id}: {exc}") from exc


def _model_ids(base_url: str) -> list[str]:
    """Every model the router knows. Asking also makes it fold a finished download into its list."""
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise CacheRouterError(f"could not list llama-server's models: {exc}") from exc
    entries = body.get("data", []) if isinstance(body, dict) else []
    return [entry.get("id") for entry in entries if isinstance(entry, dict) and entry.get("id")]


def _is_cached(model_id: str, ids: list[str]) -> bool:
    # A bare `repo` is resolved to one of its quants by llama.cpp, and cached under `repo:QUANT`.
    return model_id in ids or (":" not in model_id and any(i.startswith(model_id + ":") for i in ids))


def _stream_events(
    base_url: str, events: "queue.Queue[dict | None]", connected: threading.Event,
) -> None:
    """Feeds the router's event stream into `events`, ending with None when the stream does.
    `connected` is set once the router has accepted the subscription -- it sends its headers
    straight away, before any event."""
    try:
        with urllib.request.urlopen(f"{base_url}/models/sse", timeout=None) as resp:
            connected.set()
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    events.put(json.loads(line[len("data:"):]))
                except json.JSONDecodeError:
                    continue
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
        pass
    finally:
        connected.set()  # unblock the caller either way; the None below tells it the stream ended
        events.put(None)


def _totals(progress: dict) -> tuple[int, int | None]:
    """Sums a router progress map ({url: {"done": n, "total": n}}) across files. The total is None
    until every file has reported one -- a model with a vision projector downloads two."""
    done, total, known = 0, 0, True
    for entry in progress.values():
        if not isinstance(entry, dict):
            continue
        done += int(entry.get("done") or 0)
        file_total = int(entry.get("total") or 0)
        if file_total <= 0:
            known = False
        total += file_total
    return done, (total if known and total > 0 else None)


def download(
    server_binary: Path, model_id: str, *, timeout: float,
    on_progress: DownloadProgress | None = None,
) -> None:
    """Downloads `model_id` (a Hugging Face `repo` or `repo:quant`) into the cache, reporting bytes
    received as it goes. Returns at once if it is already cached. Raises CacheRouterError when the
    download fails, times out, or finishes without the model actually landing in the cache.

    On a timeout the scratch router is stopped, which stops its download child too; the partial
    file it leaves is resumed by the next attempt."""
    with scratch_router(server_binary) as router:
        if _is_cached(model_id, _model_ids(router.base_url)):
            log.info("%s is already in the cache", model_id)
            return

        # Subscribe before asking, so not even the first event can be missed.
        events: "queue.Queue[dict | None]" = queue.Queue()
        connected = threading.Event()
        threading.Thread(
            target=_stream_events, args=(router.base_url, events, connected),
            name="cache-router-sse", daemon=True,
        ).start()
        if not connected.wait(timeout=REQUEST_TIMEOUT_SECONDS):
            raise CacheRouterError("could not subscribe to llama-server's download events")

        request = urllib.request.Request(
            f"{router.base_url}/models",
            data=json.dumps({"model": model_id}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS):
                pass
        except urllib.error.HTTPError as exc:
            raise CacheRouterError(error_message(exc)) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise CacheRouterError(f"could not start the download of {model_id}: {exc}") from exc

        deadline = time.monotonic() + timeout
        outcome = None
        while outcome is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CacheRouterError(f"timed out after {timeout:.0f}s downloading {model_id}")
            try:
                event = events.get(timeout=min(remaining, 5.0))
            except queue.Empty:
                continue
            if event is None:
                raise CacheRouterError(f"lost llama-server's event stream while downloading {model_id}")
            if event.get("model") != model_id:
                continue
            kind = event.get("event")
            if kind == "download_progress" and on_progress is not None:
                progress = (event.get("data") or {}).get("progress") or {}
                on_progress(*_totals(progress))
            elif kind in ("download_finished", "download_failed"):
                outcome = kind

        if outcome == "download_finished" and _is_cached(model_id, _model_ids(router.base_url)):
            return
        reason = router.last_error()
        raise CacheRouterError(
            f"nothing was downloaded for {model_id} -- check the repo and quant name"
            + (f" (llama-server: {reason})" if reason else "")
        )
