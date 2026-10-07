"""aipotluck.service.jobs -- the queue behind the long-running API operations.

The property that matters most here is serialization. It is not a throughput compromise: a
benchmark taken while something else is loading a model measures the contention rather than the
model, and two models resident at once was an outright allocation failure on a 16GB board. Parallel
jobs would reintroduce both, and silently -- the scores would just be wrong.
"""

from __future__ import annotations

import threading
import time

import pytest

from aipotluck.service import jobs as jobs_module


@pytest.fixture
def runner():
    runner = jobs_module.JobRunner()
    runner.start()
    yield runner
    runner.stop(timeout=2)


def _wait_for(runner, job_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = runner.get(job_id)
        if job and job.state in jobs_module.TERMINAL_STATES:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


class TestJobLifecycle:
    def test_a_submitted_job_is_queued_immediately_and_carries_an_id(self, runner):
        """submit() must return before the work does -- that is the whole reason this exists."""
        job = runner.submit("pull", "org/m:Q4_K_M", lambda j: {"ok": True})
        assert job.id and job.state in (jobs_module.QUEUED, jobs_module.RUNNING)

    def test_a_successful_job_carries_its_result(self, runner):
        job = runner.submit("pull", "org/m:Q4_K_M", lambda j: {"score": 412})
        assert _wait_for(runner, job.id).result == {"score": 412}

    def test_a_failing_job_is_recorded_rather_than_crashing_the_worker(self, runner):
        """One bad job must not take the queue down with it, or a single failure would silently
        end model management for the lifetime of the service."""
        bad = runner.submit("pull", "org/m:Q4_K_M", lambda j: 1 / 0)
        assert _wait_for(runner, bad.id).state == jobs_module.FAILED

        good = runner.submit("pull", "org/other:Q8_0", lambda j: {"ok": True})
        assert _wait_for(runner, good.id).state == jobs_module.SUCCEEDED

    def test_a_model_op_errors_code_survives_onto_the_job(self, runner):
        """The HTTP layer and a client both branch on the code, so it has to outlive the worker
        boundary rather than being flattened into a message."""
        from aipotluck.service.model_ops import ModelOpError

        def _raise(job):
            raise ModelOpError("nope", code="download_failed", status=502)

        job = _wait_for(runner, runner.submit("pull", "m", _raise).id)
        assert job.error_code == "download_failed" and job.error == "nope"

    def test_progress_is_visible_while_the_job_is_still_running(self, runner):
        """Progress that only appeared at the end would be useless -- the point is watching a
        ten-minute pull."""
        seen = threading.Event()
        release = threading.Event()

        def _work(job):
            runner.progress_callback(job)("downloading")
            seen.set()
            release.wait(timeout=3)
            return {}

        job = runner.submit("pull", "m", _work)
        assert seen.wait(timeout=3)
        mid = runner.get(job.id)
        assert mid.state == jobs_module.RUNNING
        assert [p["message"] for p in mid.progress] == ["downloading"]
        release.set()
        _wait_for(runner, job.id)


class TestSerialization:
    def test_jobs_never_overlap(self, runner):
        """The contract a correct measurement depends on."""
        concurrent = []
        active = []
        lock = threading.Lock()

        def _work(job):
            with lock:
                active.append(1)
                concurrent.append(len(active))
            time.sleep(0.05)
            with lock:
                active.pop()
            return {}

        ids = [runner.submit("benchmark", f"m{i}", _work).id for i in range(4)]
        for job_id in ids:
            _wait_for(runner, job_id)
        assert max(concurrent) == 1, "two jobs ran at once; measurements would be meaningless"

    def test_queued_work_still_runs_in_submission_order(self, runner):
        order = []
        ids = [
            runner.submit("benchmark", f"m{i}", lambda j, i=i: order.append(i) or {}).id
            for i in range(4)
        ]
        for job_id in ids:
            _wait_for(runner, job_id)
        assert order == [0, 1, 2, 3]


class TestListingAndRetention:
    def test_listing_is_newest_first(self, runner):
        """A caller polling for "what just happened" should not page through history for it."""
        first = runner.submit("pull", "a", lambda j: {})
        second = runner.submit("pull", "b", lambda j: {})
        _wait_for(runner, second.id)
        assert [j.id for j in runner.list()][:2] == [second.id, first.id]

    def test_finished_jobs_are_pruned_but_recent_ones_survive(self, runner):
        ids = [runner.submit("pull", f"m{i}", lambda j: {}).id for i in range(
            jobs_module.MAX_FINISHED_RETAINED + 5
        )]
        _wait_for(runner, ids[-1], timeout=10)
        assert runner.get(ids[-1]) is not None, "the newest result must still be collectable"
        assert runner.get(ids[0]) is None, "the oldest should have been pruned"
        assert len(runner.list()) <= jobs_module.MAX_FINISHED_RETAINED

    def test_an_unknown_job_id_is_none_rather_than_an_error(self, runner):
        assert runner.get("nope") is None


class TestDownloadProgress:
    def test_the_latest_byte_count_replaces_the_last_rather_than_piling_up(self):
        """Updates arrive several times a second; a list of them would grow by thousands of
        entries over a large pull and make every poll of the job heavier."""
        runner = jobs_module.JobRunner()
        job = runner.submit("pull", "org/m:Q4_K_M", lambda job: {})
        report = runner.download_callback(job)

        report(10, None)
        report(500, 1000)

        snapshot = job.to_dict()
        assert snapshot["download"]["received_bytes"] == 500
        assert snapshot["download"]["total_bytes"] == 1000
        assert snapshot["progress"] == []

    def test_a_job_that_never_downloads_reports_none(self):
        runner = jobs_module.JobRunner()
        job = runner.submit("benchmark", None, lambda job: {})
        assert job.to_dict()["download"] is None
