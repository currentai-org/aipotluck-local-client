"""The HTTP model-management API.

Real requests against a real server throughout -- these prove the routing, the status codes and the
JSON shapes a remote caller actually sees, which is the part a unit test of model_ops cannot reach.
The operations themselves are stubbed at the model_ops boundary, because what is under test here is
the API surface, not the measurement machinery (tests/test_model_* cover that).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pytest

from aipotluck.service import jobs as jobs_module
from aipotluck.service import model_ops, runner


@pytest.fixture
def service(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "runtime.json").write_text(
        json.dumps({
            "logged_in": False,
            "llama_cpp": {"server_binary": "/fake/llama-server", "presets_path": str(tmp_path / "p.ini")},
        }),
        encoding="utf-8",
    )
    svc = runner.AipotluckServiceRunner(
        host="127.0.0.1", port=0, config_dir=config_dir, log_dir=None
    )
    svc.start()
    try:
        yield svc, f"http://127.0.0.1:{svc._server.server_address[1]}"
    finally:
        svc.stop(timeout=3)


def _request(url, *, method="GET", body=None):
    """Returns (status, parsed-json). An error status is a response here, not an exception."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"null")


def _await_job(base, job_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _status, job = _request(f"{base}/jobs/{job_id}")
        if job["state"] in jobs_module.TERMINAL_STATES:
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


class TestListModels:
    def test_returns_the_shared_listing(self, service, monkeypatch):
        svc, base = service
        monkeypatch.setattr(
            model_ops, "list_models",
            lambda cfg, cd: {"models": [{"id": "org/m:Q4_K_M"}], "engine": {"tag": "b1"}},
        )
        status, body = _request(f"{base}/models")
        assert status == 200
        assert body["models"][0]["id"] == "org/m:Q4_K_M"

    def test_a_model_op_error_becomes_its_own_status_and_code(self, service, monkeypatch):
        """The status is decided once, in model_ops, so the CLI and the API cannot disagree about
        whether something is a 404 or a 503."""
        svc, base = service

        def _boom(cfg, cd):
            raise model_ops.ModelOpError("nope", code="cache_unreadable", status=503)

        monkeypatch.setattr(model_ops, "list_models", _boom)
        status, body = _request(f"{base}/models")
        assert status == 503 and body["error"]["code"] == "cache_unreadable"


class TestPull:
    def test_a_pull_returns_202_and_a_job_to_poll(self, service, monkeypatch):
        """The request must return before the work does: a real pull is minutes long and no client
        will hold a connection open for it through a tunnel."""
        svc, base = service
        monkeypatch.setattr(model_ops, "check_size_before_download", lambda m: None)
        monkeypatch.setattr(model_ops, "pull", lambda *a, **kw: {"model": kw.get("model") or a[2]})

        status, body = _request(f"{base}/models", method="POST", body={"model": "org/m:Q4_K_M"})

        assert status == 202
        assert body["job"]["kind"] == "pull" and body["job"]["model"] == "org/m:Q4_K_M"
        assert _await_job(base, body["job"]["id"])["state"] == jobs_module.SUCCEEDED

    def test_an_oversized_model_is_refused_before_the_job_is_queued(self, service, monkeypatch):
        """Refusing in the response the caller is already waiting on, rather than as a job that
        fails minutes later, is the whole value of a pre-download check."""
        svc, base = service
        monkeypatch.setattr(
            model_ops, "check_size_before_download",
            lambda m: {"fits": False, "detail": "too big", "model_bytes": 9, "total_memory_bytes": 1},
        )
        submitted = []
        monkeypatch.setattr(model_ops, "pull", lambda *a, **kw: submitted.append(1) or {})

        status, body = _request(f"{base}/models", method="POST", body={"model": "org/m:Q4_K_M"})

        assert status == 409
        assert body["error"]["code"] == "too_large_for_device"
        assert body["error"]["retry_with"] == {"allow_oversized": True}
        assert submitted == [], "nothing should have been queued"

    def test_allow_oversized_skips_the_check_and_queues_the_pull(self, service, monkeypatch):
        svc, base = service
        checks = []
        monkeypatch.setattr(
            model_ops, "check_size_before_download",
            lambda m: checks.append(1) or {"fits": False, "detail": "too big"},
        )
        monkeypatch.setattr(model_ops, "pull", lambda *a, **kw: {})
        status, body = _request(
            f"{base}/models", method="POST",
            body={"model": "org/m:Q4_K_M", "allow_oversized": True},
        )
        assert status == 202 and checks == []

    def test_the_two_overrides_reach_the_operation_separately(self, service, monkeypatch):
        """allow_oversized and keep_rejected answer different questions, and a caller overriding
        the cheap pre-check must not silently also disable the measured verdict."""
        svc, base = service
        seen = {}
        monkeypatch.setattr(model_ops, "check_size_before_download", lambda m: None)
        monkeypatch.setattr(model_ops, "pull", lambda *a, **kw: seen.update(kw) or {})

        _request(f"{base}/models", method="POST",
                 body={"model": "org/m:Q4_K_M", "allow_oversized": True})
        _await_job(base, _request(f"{base}/jobs")[1]["jobs"][0]["id"])

        assert seen["keep_rejected"] is False

    def test_force_sets_both_overrides(self, service, monkeypatch):
        svc, base = service
        seen = {}
        monkeypatch.setattr(model_ops, "check_size_before_download", lambda m: None)
        monkeypatch.setattr(model_ops, "pull", lambda *a, **kw: seen.update(kw) or {})

        body = _request(f"{base}/models", method="POST",
                        body={"model": "org/m:Q4_K_M", "force": True})[1]
        _await_job(base, body["job"]["id"])

        assert seen["keep_rejected"] is True

    def test_a_missing_model_field_is_a_400(self, service):
        svc, base = service
        status, body = _request(f"{base}/models", method="POST", body={})
        assert status == 400 and body["error"]["code"] == "bad_request"

    def test_an_unparseable_body_is_a_400_rather_than_a_500(self, service):
        svc, base = service
        req = urllib.request.Request(
            f"{base}/models", data=b"{not json", method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=10)
            raise AssertionError("expected an error status")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400

    def test_a_failing_pull_surfaces_on_the_job_with_its_code(self, service, monkeypatch):
        svc, base = service
        monkeypatch.setattr(model_ops, "check_size_before_download", lambda m: None)

        def _boom(*a, **kw):
            raise model_ops.ModelOpError("hf down", code="download_failed", status=502)

        monkeypatch.setattr(model_ops, "pull", _boom)
        body = _request(f"{base}/models", method="POST", body={"model": "org/m:Q4_K_M"})[1]
        job = _await_job(base, body["job"]["id"])
        assert job["state"] == jobs_module.FAILED and job["error_code"] == "download_failed"


class TestBenchmark:
    def test_benchmarking_one_model_returns_a_job(self, service, monkeypatch):
        svc, base = service
        monkeypatch.setattr(model_ops, "benchmark", lambda *a, **kw: {"results": [], "measured": 0})
        status, body = _request(
            f"{base}/models/benchmark", method="POST", body={"model": "org/m:Q4_K_M"}
        )
        assert status == 202 and body["job"]["model"] == "org/m:Q4_K_M"

    def test_an_empty_body_means_measure_everything(self, service, monkeypatch):
        """`benchmark` with no model is the CLI's "measure every cached model", and the API has to
        express the same thing rather than demanding one."""
        svc, base = service
        seen = []
        monkeypatch.setattr(
            model_ops, "benchmark",
            lambda cfg, cd, model_id=None, **kw: seen.append(model_id) or {"results": []},
        )
        status, body = _request(f"{base}/models/benchmark", method="POST")
        assert status == 202
        _await_job(base, body["job"]["id"])
        assert seen == [None]

    def test_progress_is_readable_while_the_job_runs(self, service, monkeypatch):
        svc, base = service

        def _slow(cfg, cd, model_id=None, progress=None, **kw):
            progress("screening org/m:Q4_K_M for viability")
            return {"results": []}

        monkeypatch.setattr(model_ops, "benchmark", _slow)
        body = _request(f"{base}/models/benchmark", method="POST", body={"model": "org/m:Q4_K_M"})[1]
        job = _await_job(base, body["job"]["id"])
        assert [p["message"] for p in job["progress"]] == ["screening org/m:Q4_K_M for viability"]


class TestRemove:
    def test_deleting_by_query_parameter(self, service, monkeypatch):
        """?model= rather than a path segment: a model id carries both "/" and ":", and this is the
        shape llama.cpp's own router uses for the same operation."""
        svc, base = service
        seen = []
        monkeypatch.setattr(
            model_ops, "remove",
            lambda cfg, cd, model_id: seen.append(model_id) or {"model": model_id, "removed": True},
        )
        status, body = _request(
            f"{base}/models?model=org%2Fm%3AQ4_K_M", method="DELETE"
        )
        assert status == 200 and body["removed"] is True
        assert seen == ["org/m:Q4_K_M"], "the id must survive url-encoding intact"

    def test_removal_is_synchronous(self, service, monkeypatch):
        """It finishes in milliseconds, so making the caller poll a job for it would be ceremony
        with no benefit."""
        svc, base = service
        monkeypatch.setattr(model_ops, "remove", lambda *a: {"model": "m", "removed": True})
        status, body = _request(f"{base}/models?model=m", method="DELETE")
        assert status == 200 and "job" not in body

    def test_a_missing_model_parameter_is_a_400(self, service):
        svc, base = service
        status, body = _request(f"{base}/models", method="DELETE")
        assert status == 400 and body["error"]["code"] == "bad_request"

    def test_an_unknown_model_is_a_404(self, service, monkeypatch):
        svc, base = service

        def _boom(cfg, cd, model_id):
            raise model_ops.ModelOpError("nope", code="unknown_model", status=404)

        monkeypatch.setattr(model_ops, "remove", _boom)
        status, body = _request(f"{base}/models?model=org%2Fm", method="DELETE")
        assert status == 404 and body["error"]["code"] == "unknown_model"


class TestJobsEndpoint:
    def test_an_unknown_job_is_a_404(self, service):
        svc, base = service
        status, body = _request(f"{base}/jobs/nope")
        assert status == 404 and body["error"]["code"] == "unknown_job"

    def test_listing_jobs(self, service, monkeypatch):
        svc, base = service
        monkeypatch.setattr(model_ops, "benchmark", lambda *a, **kw: {"results": []})
        _request(f"{base}/models/benchmark", method="POST", body={"model": "a"})
        status, body = _request(f"{base}/jobs")
        assert status == 200 and len(body["jobs"]) == 1

    def test_an_unknown_route_is_still_a_404(self, service):
        svc, base = service
        status, _body = _request(f"{base}/nope")
        assert status == 404


class TestExistingEndpointsStillWork:
    """The new routes must not have shadowed the ones that were already there."""

    def test_healthz(self, service):
        svc, base = service
        with urllib.request.urlopen(f"{base}/healthz", timeout=5) as resp:
            assert resp.status == 200 and resp.read() == b"ok"

    def test_status(self, service):
        svc, base = service
        status, body = _request(f"{base}/status")
        assert status == 200 and body["service"] == runner.SERVICE_NAME
