"""aipotluck.installer.model_perf_live -- the no-llama-bench fallback.

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


class TestProbePerformanceLive:
    def test_recovers_the_cost_model_from_the_servers_own_timings(self):
        with RouterStub(prefill_base=2.0, prefill_slope=0.0004, decode_base=20.0, decode_slope=0.001) as router:
            result = live.probe_performance_live(router.base_url, "org/repo:Q4_K_M", ctx_size=131072)

        assert result.fit.decode_base_ms == pytest.approx(20.0, rel=1e-3)
        assert result.fit.decode_depth_ms == pytest.approx(0.001, rel=1e-3)
        assert result.fit.prefill_base_ms == pytest.approx(2.0, rel=1e-3)
        assert result.fit.prefill_depth_ms == pytest.approx(0.0004, rel=1e-3)
        assert result.fit.source == mp.SOURCE_LIVE_SERVER

    def test_disables_prompt_caching_so_the_deep_prefill_is_actually_paid(self):
        """With cache_prompt left on, the second request would reuse the first one's prefix and
        report a prefill that never happened -- the depth slope would come back near zero and the
        model would look far better at long context than it is."""
        with RouterStub() as router:
            live.probe_performance_live(router.base_url, "org/repo:Q4_K_M", ctx_size=131072)

        completions = [body for _, path, body in router.requests if path == "/v1/chat/completions"]
        assert len(completions) == 2
        assert all(body["cache_prompt"] is False for body in completions)
        assert all(body["temperature"] == 0.0 for body in completions)
        # Two genuinely different prompt lengths, or there is no lever arm for the fit.
        lengths = [len(body["messages"][0]["content"].split()) for body in completions]
        assert lengths[1] > lengths[0] * 4

    def test_unloads_first_so_the_cold_load_is_measurable(self):
        """Cold load sits inside the web app's turn budget and is the common case under
        --models-max 1, so it is measured rather than assumed away."""
        with RouterStub() as router:
            live.probe_performance_live(router.base_url, "org/repo:Q4_K_M", ctx_size=131072)

        assert router.requests[0][1] == "/models/unload"

    def test_a_hopeless_model_is_refused_after_a_single_request(self):
        # Slow enough that even the optimistic shallow-only fit cannot reach OUTPUT_RED_MIN.
        with RouterStub(decode_base=900.0) as router:
            result = live.probe_performance_live(router.base_url, "org/repo:Q4_K_M", ctx_size=131072)

        assert result.grade == mp.GRADE_REFUSE
        assert len([r for r in router.requests if r[1] == "/v1/chat/completions"]) == 1

    def test_a_build_without_timings_raises_rather_than_guessing(self):
        with RouterStub(omit_timings=True) as router:
            with pytest.raises(mp.ModelPerfError, match="timings"):
                live.probe_performance_live(router.base_url, "org/repo:Q4_K_M", ctx_size=131072)

    def test_an_unreachable_router_raises(self):
        with pytest.raises(mp.ModelPerfError, match="could not reach"):
            live.probe_performance_live("http://127.0.0.1:1", "org/repo:Q4_K_M", ctx_size=4096)
