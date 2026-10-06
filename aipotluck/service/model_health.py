"""Catching, diagnosing and recovering from a model that runs out of memory at runtime.

`model_preflight` and `model_sizing` both judge a model before it ever serves a turn, from numbers
that are stable: how big the weights are, how much memory the machine has. This module exists
because that is not the whole story. The memory actually free when a model loads depends on what
else the user is doing, and that moves -- a browser, a build, another model's leftovers. A context
that was correctly sized on a quiet machine can be too large an hour later, and llama.cpp's answer
to "too large" is to die.

So an OOM here is **not** automatically a verdict about the model. Treating one as proof the model
is unusable would evict perfectly good models whenever the user opened something heavy. The policy
is therefore: shrink and retry, and only conclude the model is the problem when it keeps happening
without a working turn in between.

## What we can observe

Router mode spawns a separate llama-server child per model (server-models.cpp), so a load or
inference OOM kills that child and leaves the router itself running. Two surfaces report it:

* `GET /models` gives structured per-model state -- `status.failed` and `status.exit_code` once a
  child has died. This is the trigger: cheap, polled, and unambiguous about *that* a failure
  happened.
* The router forwards every child's combined stdout/stderr into its own log, prefixed with the
  child's port (`subprocess_option_combined_stdout_stderr` at server-models.cpp's spawn). This is
  what says *why*, because an exit code cannot: a C++ allocation throw, an OOM-killer SIGKILL and a
  corrupt GGUF produce overlapping codes.

Both are needed. Acting on the exit code alone would shrink the context of a model that failed for
reasons context has nothing to do with, which does not fix it and quietly degrades it.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from aipotluck.installer import llama_errors, model_presets, model_sizing

log = logging.getLogger("aipotluck.service.model_health")

# Recognising llama.cpp's failures is shared with the installer's sizing probe, which hits exactly
# the same wall from the other side -- see aipotluck.installer.llama_errors for the signatures, the
# exit codes, and why a bounded excerpt has to drop the backtrace to keep the error.
OOM_LOG_SIGNATURES = llama_errors.OOM_LOG_SIGNATURES
OOM_EXIT_CODES = llama_errors.OOM_EXIT_CODES
FAILURE_OOM = llama_errors.FAILURE_OOM
FAILURE_OTHER = llama_errors.FAILURE_OTHER
classify_failure = llama_errors.classify_failure

# How many OOMs a model may hit, with no working turn in between, before we stop blaming the
# machine and start blaming the model. Three gives two real shrink-and-retry attempts before the
# verdict, which is enough to ride out a transient spike without looping forever on a model that
# genuinely cannot run here.
MAX_OOMS_BEFORE_QUARANTINE = 3

# Each recovery attempt halves the context. Halving rather than shaving is deliberate: the KV cache
# is linear in context, so a 10% trim buys 10% of the headroom that was just proven insufficient,
# and a model would exhaust its retries still failing.
CONTEXT_SHRINK_DIVISOR = 2

# Router log lines. The router announces which port it proxied a model to; each child's own lines
# then carry that port as a prefix. Together they map a success line back to a model id.
_PROXY_RE = re.compile(r"proxying request to model (\S+) on port (\d+)")
_CHILD_LINE_RE = re.compile(r"^\[(\d+)\]\s*(.*)$")

# What a completed turn looks like in a child's log. `release: ... stop processing` is printed once
# the slot is done with a task, which is the narrowest "this model actually served something"
# marker available without instrumenting the inference path itself.
_TURN_DONE_RE = re.compile(r"release:.*stop processing")


def models_that_served(log_chunk: str) -> set[str]:
    """Model ids that completed a turn somewhere in `log_chunk`.

    Resolved through the child's port rather than by proximity: the router interleaves lines from
    every child it is proxying for, so "the last model mentioned" is not reliably the model whose
    turn just finished. The port prefix is."""
    port_to_model: dict[str, str] = {}
    served: set[str] = set()

    for line in log_chunk.splitlines():
        proxied = _PROXY_RE.search(line)
        if proxied:
            port_to_model[proxied.group(2)] = proxied.group(1)
            continue
        child = _CHILD_LINE_RE.match(line)
        if child and _TURN_DONE_RE.search(child.group(2)):
            model_id = port_to_model.get(child.group(1))
            if model_id:
                served.add(model_id)
    return served


def shrunk_context(current_ctx: int, *, floor: int) -> int | None:
    """The context to retry at after an OOM, or None when there is no useful room left.

    Returning None is the signal that shrinking has stopped being a recovery and started being a
    way to produce a model that loads but cannot hold a conversation. `floor` is the smallest
    context that can serve a turn, so landing under it is a rejection, not a smaller success."""
    if current_ctx <= floor:
        return None
    candidate = current_ctx // CONTEXT_SHRINK_DIVISOR
    if candidate < floor:
        return None
    return candidate


# ---------------------------------------------------------------------------
# Persistence
#
# The OOM count has to survive a service restart. A model that OOMs hard enough to take the router
# down with it would otherwise reset its own counter on the way back up, and the restart loop would
# retry forever -- which is precisely the "slips through the cracks" case this exists to end.
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1
_HEALTH_FILENAME = "model_health.json"


def health_path(config_dir: Path) -> Path:
    return Path(config_dir) / _HEALTH_FILENAME


def read_all(config_dir: Path) -> dict[str, Any]:
    path = health_path(config_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        return {}
    return data


def read_record(config_dir: Path, model_id: str) -> dict[str, Any] | None:
    record = read_all(config_dir).get("models", {}).get(model_id)
    return record if isinstance(record, dict) else None


def quarantined_models(config_dir: Path) -> dict[str, dict[str, Any]]:
    """Every model currently held back, for `list` and `/status` to report."""
    models = read_all(config_dir).get("models", {})
    if not isinstance(models, dict):
        return {}
    return {
        model_id: record
        for model_id, record in models.items()
        if isinstance(record, dict) and record.get("quarantined")
    }


def _write(config_dir: Path, data: dict[str, Any]) -> None:
    path = health_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data["schema_version"] = SCHEMA_VERSION
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _mutate(config_dir: Path, model_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read-modify-write scaffolding: another model's record is never touched by a write for this
    one (same posture as model_perf_store.write_record)."""
    data = read_all(config_dir)
    models = data.setdefault("models", {})
    if not isinstance(models, dict):
        models = {}
        data["models"] = models
    record = models.setdefault(model_id, {})
    if not isinstance(record, dict):
        record = {}
        models[model_id] = record
    return data, record


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def record_oom(config_dir: Path, model_id: str, *, from_ctx: int | None, to_ctx: int | None) -> int:
    """Counts one OOM and returns the running total since the model last served a turn."""
    data, record = _mutate(config_dir, model_id)
    record["oom_count"] = int(record.get("oom_count") or 0) + 1
    record["last_oom_at"] = _now_iso()
    record["last_ctx_size"] = to_ctx if to_ctx is not None else from_ctx
    history = record.setdefault("shrink_history", [])
    if isinstance(history, list) and from_ctx is not None:
        history.append({"at": record["last_oom_at"], "from": from_ctx, "to": to_ctx})
        del history[:-10]  # a bounded trail; this file is a diagnostic, not an audit log
    _write(config_dir, data)
    return record["oom_count"]


def record_success(config_dir: Path, model_id: str) -> None:
    """A completed turn clears the OOM count.

    This is what makes the policy "three OOMs without a working turn" rather than "three OOMs
    ever". A model that serves fine for a week and then hits a memory spike starts from zero, so
    normal use can never accumulate its way into a quarantine."""
    data = read_all(config_dir)
    models = data.get("models")
    if not isinstance(models, dict):
        return
    record = models.get(model_id)
    if not isinstance(record, dict) or not record.get("oom_count"):
        return
    if record.get("quarantined"):
        return  # a quarantined model's history is kept until the user decides what to do with it
    record["oom_count"] = 0
    record["last_success_at"] = _now_iso()
    _write(config_dir, data)


def quarantine(config_dir: Path, model_id: str, reason: str) -> None:
    data, record = _mutate(config_dir, model_id)
    record["quarantined"] = True
    record["quarantine_reason"] = reason
    record["quarantined_at"] = _now_iso()
    _write(config_dir, data)


def forget(config_dir: Path, model_id: str) -> None:
    """Drops a model's health history -- on removal, or on a deliberate re-pull, so the model gets
    a clean slate rather than inheriting a verdict reached against a different sizing."""
    data = read_all(config_dir)
    models = data.get("models")
    if not isinstance(models, dict) or model_id not in models:
        return
    del models[model_id]
    _write(config_dir, data)


# ---------------------------------------------------------------------------
# The watcher
# ---------------------------------------------------------------------------

# How long to wait before reacting to the same model's failure again. The router keeps reporting
# `failed` until something loads successfully, so without this one dead child would be counted on
# every poll and a model would burn through its whole retry budget in a few seconds. It also paces
# the retries themselves: a memory spike the user caused by opening something heavy is often over
# within a minute, and the next attempt should land after it, not during it.
REACT_COOLDOWN_SECONDS = 45.0

DEFAULT_POLL_INTERVAL_SECONDS = 10.0
_HTTP_TIMEOUT_SECONDS = 5.0

# Never read an unbounded amount of a log that has been growing while we were not looking.
_MAX_LOG_CHUNK_BYTES = 2 * 1024 * 1024


class ModelHealthWatcher:
    """Polls the router for dead model instances and tries to get them running again.

    Runs as its own daemon thread beside LlamaSupervisor rather than inside it, because the two
    supervise different things: LlamaSupervisor keeps the router *process* alive, while this keeps
    the *models* the router serves usable. A model child dying does not bring the router down, so
    nothing in the existing supervision path would ever notice it.

    Every tick is wrapped: a watcher that throws must not take the service with it, since the
    service's actual job -- serving turns -- works fine without any of this.
    """

    def __init__(
        self,
        *,
        base_url: str,
        presets_path: Path,
        config_dir: Path,
        log_path: Path | None,
        poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
        min_ctx_tokens: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.presets_path = Path(presets_path)
        self.config_dir = Path(config_dir)
        self.log_path = Path(log_path) if log_path else None
        self.poll_interval = poll_interval
        self.min_ctx_tokens = (
            min_ctx_tokens if min_ctx_tokens is not None else model_sizing.MIN_SERVABLE_CTX_TOKENS
        )

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._log_offset = 0
        self._log_inode: int | None = None
        self._last_reacted_at: dict[str, float] = {}

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        # Start from the end of the log: lines written before this service started describe a
        # previous run, and replaying them would credit or blame models for history we already
        # accounted for.
        self._seek_to_end()
        self._thread = threading.Thread(target=self._run, name="aipotluck-model-health", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self.poll_interval):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 -- see the class docstring
                log.exception("Model health check failed; the service continues without it")

    # -- the log -----------------------------------------------------------

    def _seek_to_end(self) -> None:
        if not self.log_path:
            return
        try:
            stat = self.log_path.stat()
        except OSError:
            return
        self._log_offset = stat.st_size
        self._log_inode = stat.st_ino

    def _read_new_log(self) -> str:
        """Whatever the router has written since the last read.

        Handles rotation two ways, because both happen: a new inode means the file was replaced,
        and a size smaller than our offset means it was truncated in place. Either way the right
        move is to start from the beginning of what is now there rather than seek past the end."""
        if not self.log_path:
            return ""
        try:
            stat = self.log_path.stat()
        except OSError:
            return ""

        if self._log_inode is not None and stat.st_ino != self._log_inode:
            self._log_offset = 0
        elif stat.st_size < self._log_offset:
            self._log_offset = 0
        self._log_inode = stat.st_ino

        if stat.st_size <= self._log_offset:
            return ""
        start = max(self._log_offset, stat.st_size - _MAX_LOG_CHUNK_BYTES)
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(start)
                chunk = fh.read()
        except OSError:
            return ""
        self._log_offset = stat.st_size
        return chunk

    # -- the router --------------------------------------------------------

    def _fetch_models(self) -> list[dict]:
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/models", timeout=_HTTP_TIMEOUT_SECONDS
            ) as response:
                payload = json.load(response)
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
            log.debug("Could not read the router's model list: %s", exc)
            return []
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            models = payload.get("data") if isinstance(payload, dict) else None
        return [m for m in (models or []) if isinstance(m, dict)]

    def _request_reload(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/models?reload=1", timeout=_HTTP_TIMEOUT_SECONDS
            ) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError):
            return False

    # -- the work ----------------------------------------------------------

    def tick(self) -> None:
        chunk = self._read_new_log()

        # Successes first. A model that both served and failed within one interval has recovered,
        # and crediting the success before counting the failure is the forgiving order -- which is
        # the right bias for a policy whose whole point is not to evict working models.
        for model_id in models_that_served(chunk):
            record_success(self.config_dir, model_id)

        for model in self._fetch_models():
            status = model.get("status")
            if not isinstance(status, dict) or not status.get("failed"):
                continue
            model_id = model.get("id")
            if not model_id:
                continue
            self._handle_failure(model_id, status.get("exit_code"), chunk)

    def _handle_failure(self, model_id: str, exit_code: int | None, log_chunk: str) -> None:
        now = time.monotonic()
        last = self._last_reacted_at.get(model_id)
        if last is not None and now - last < REACT_COOLDOWN_SECONDS:
            return

        record = read_record(self.config_dir, model_id) or {}
        if record.get("quarantined"):
            return  # already decided; the user owns what happens next

        kind = classify_failure(log_chunk, exit_code)
        self._last_reacted_at[model_id] = now
        if kind is not FAILURE_OOM:
            # Shrinking the context of a model that failed for an unrelated reason would not fix
            # it and would quietly degrade it, so this is reported and left alone.
            log.warning(
                "%s failed to load or run (exit_code=%s) and it does not look like an "
                "out-of-memory, so its context is left alone.", model_id, exit_code,
            )
            return

        preset = model_presets.read_all(self.presets_path).get(model_id) or {}
        try:
            current_ctx = int(preset.get("ctx-size"))
        except (TypeError, ValueError):
            current_ctx = None

        new_ctx = (
            shrunk_context(current_ctx, floor=self.min_ctx_tokens)
            if current_ctx is not None else None
        )
        count = record_oom(self.config_dir, model_id, from_ctx=current_ctx, to_ctx=new_ctx)

        # The streak is the ONLY thing that quarantines a model, and it resets on every completed
        # turn. Nothing else may, because every other condition here is a statement about the
        # machine right now. An earlier version also quarantined on reaching the context floor,
        # which looked reasonable and was wrong: a model serving perfectly well between a user's
        # occasional memory spikes would shrink a step per spike and eventually be held back for
        # having survived four of them.
        if count >= MAX_OOMS_BEFORE_QUARANTINE:
            self._quarantine(
                model_id,
                f"ran out of memory {count} times without completing a turn in between, most "
                f"recently at a {current_ctx}-token context",
            )
            return

        if new_ctx is None:
            # No remedy available this round. The model stays exactly as it is: if this is a
            # passing squeeze it will serve again and the count clears, and if it is not, the
            # streak above reaches the limit on its own.
            log.warning(
                "%s ran out of memory at a %s-token context, already the smallest that can hold a "
                "conversation -- there is nothing left to reduce (%d of %d before it is held back)",
                model_id, current_ctx, count, MAX_OOMS_BEFORE_QUARANTINE,
            )
            return

        preset["ctx-size"] = str(new_ctx)
        model_presets.write_preset(self.presets_path, model_id, preset)
        model_presets.write_tuning(self.presets_path, model_id, {
            "ctx_size": (
                f"reduced from {current_ctx} to {new_ctx} after running out of memory while "
                f"loading or serving (attempt {count} of {MAX_OOMS_BEFORE_QUARANTINE}). Re-pull "
                "the model to size it again from scratch."
            ),
        })
        reloaded = self._request_reload()
        log.warning(
            "%s ran out of memory (%d of %d); retrying with a %d-token context instead of %d%s",
            model_id, count, MAX_OOMS_BEFORE_QUARANTINE, new_ctx, current_ctx,
            "" if reloaded else " (the router did not confirm the reload)",
        )

    def _quarantine(self, model_id: str, reason: str) -> None:
        quarantine(self.config_dir, model_id, reason)
        log.error(
            "%s has been held back: %s. It is still on disk -- remove it with "
            "`aipotluck-local-client remove %s`, or re-pull it to try again.",
            model_id, reason, model_id,
        )
