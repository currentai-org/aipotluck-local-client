"""Background jobs for the operations that take minutes rather than milliseconds.

A pull downloads several gigabytes, sizes the model, screens it and scores it; on a Jetson that is
comfortably past ten minutes. No HTTP client will hold a connection open that long through a
tunnel, and if one drops there is no way to learn whether the work finished, so the request has to
return before the work does.

**Jobs run one at a time, deliberately.** This is not a throughput compromise -- it is what makes
the measurements mean anything. `model_perf.wait_until_idle` exists because a benchmark taken on a
busy machine describes the contention rather than the model, and `model_ops.measure` frees whatever
the router is holding before it starts, because two models resident at once was an outright
allocation failure on a 16GB board. Two jobs in parallel would reintroduce both problems and the
second one silently: the scores would simply be wrong, with nothing to say so.

So a submitted job is queued, and the queue is part of the contract the API reports.
"""

from __future__ import annotations

import datetime
import logging
import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("aipotluck.service.jobs")

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"

TERMINAL_STATES = (SUCCEEDED, FAILED)

# Finished jobs are kept so a caller that reconnects can still collect a result it missed, but not
# forever -- this is a long-lived service on a small device.
MAX_FINISHED_RETAINED = 50


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    kind: str
    model: str | None
    state: str = QUEUED
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    progress: list[dict[str, str]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "model": self.model,
            "state": self.state,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": list(self.progress),
            "result": self.result,
            "error": self.error,
            "error_code": self.error_code,
        }


class JobRunner:
    """A single-worker queue. Start once; stop() is safe to call more than once."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._queue: "queue.Queue[tuple[Job, Callable[[Job], dict[str, Any]]] | None]" = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="aipotluck-jobs", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stopping.set()
        self._queue.put(None)  # wake the worker so it can notice and exit
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def submit(self, kind: str, model: str | None, work: Callable[[Job], dict[str, Any]]) -> Job:
        """Queues `work` and returns its job immediately, already carrying an id to poll on."""
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, model=model)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
        self._queue.put((job, work))
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        """Newest first -- a caller polling for "what just happened" should not page through
        history to find it."""
        with self._lock:
            return [self._jobs[job_id] for job_id in reversed(self._order) if job_id in self._jobs]

    def _record_progress(self, job: Job, message: str) -> None:
        with self._lock:
            job.progress.append({"at": _now(), "message": message})

    def _run(self) -> None:
        while not self._stopping.is_set():
            item = self._queue.get()
            if item is None:
                continue
            job, work = item
            if self._stopping.is_set():
                # Don't start new work during shutdown; leaving it queued is honest about what
                # happened rather than reporting a failure nobody caused.
                continue
            self._execute(job, work)

    def _execute(self, job: Job, work: Callable[[Job], dict[str, Any]]) -> None:
        with self._lock:
            job.state = RUNNING
            job.started_at = _now()
        try:
            result = work(job)
        except Exception as exc:  # noqa: BLE001 -- a job's failure is data, not a crash
            log.warning("Job %s (%s) failed: %s", job.id, job.kind, exc)
            with self._lock:
                job.state = FAILED
                job.error = str(exc)
                job.error_code = getattr(exc, "code", None) or type(exc).__name__
                job.finished_at = _now()
        else:
            with self._lock:
                job.state = SUCCEEDED
                job.result = result
                job.finished_at = _now()
        self._prune()

    def _prune(self) -> None:
        """Drops the oldest finished jobs past the retention limit. Running and queued jobs are
        never dropped, however many there are -- forgetting live work would make the queue depth
        this API reports a lie."""
        with self._lock:
            finished = [
                job_id for job_id in self._order
                if job_id in self._jobs and self._jobs[job_id].state in TERMINAL_STATES
            ]
            for job_id in finished[:-MAX_FINISHED_RETAINED] if len(finished) > MAX_FINISHED_RETAINED else []:
                self._jobs.pop(job_id, None)
                self._order.remove(job_id)

    def progress_callback(self, job: Job) -> Callable[[str], None]:
        return lambda message: self._record_progress(job, message)
