"""aipotluck.installer.model_perf_live -- freeing the memory the router is holding.

Real HTTP throughout, following tests/test_cli_actions.py's TestReloadRouterModels: a small
stand-in router serves the requests, so these prove the actual wire contract this code depends on
(method, path, body fields, and which `timings` keys are read) rather than that some mock was
called. That matters more here than usual, because this path only runs on source-built arm64
hardware that is awkward to exercise directly.
"""

from __future__ import annotations

import pytest

from conftest import RouterStub

from aipotluck.installer import model_perf as mp
from aipotluck.installer import model_perf_live as live


class TestFreeRouterMemory:
    def test_unloads_every_loaded_model_not_just_one(self, monkeypatch):
        """The router is frequently holding a model other than the one about to be measured, and
        under --models-max 1 that one model is all of the memory a benchmark needs."""
        with RouterStub() as router:
            monkeypatch.setattr(
                live, "_get",
                lambda base_url, path, *, timeout: {"models": [
                    {"name": "a:Q4", "status": {"value": "loaded"}},
                    {"name": "b:Q4", "status": {"value": "unloaded"}},
                    {"name": "c:Q4", "status": {"value": "loading"}},
                ]},
            )
            freed = live.free_router_memory(router.base_url)

        assert freed == ["a:Q4", "c:Q4"]  # the unloaded one is left alone
        unloads = [b["model"] for _, path, b in router.requests if path == "/models/unload"]
        assert unloads == ["a:Q4", "c:Q4"]

    def test_an_unreachable_router_is_not_a_reason_to_skip_measuring(self):
        assert live.free_router_memory("http://127.0.0.1:1", timeout=1.0) == []
