"""aipotluck.installer.model_localscore -- LocalScore's formula over a projected cost model.

The arithmetic here is checked against two independent things: localscore.cpp's own definition
(read from source, not prose), and real measured runs on a Jetson Orin NX where the full
nine-scenario benchmark was run for comparison.
"""

from __future__ import annotations

import pytest

from aipotluck.installer import model_localscore as ls


def flat(decode_base, decode_depth=0.0, prefill_ms_per_tok=1.0):
    """A cost model whose prefill is the same at every anchor -- the shape a straight line assumes,
    useful for exercising the formula itself."""
    return ls.CostModel({16: prefill_ms_per_tok, 64: prefill_ms_per_tok, 1024: prefill_ms_per_tok},
                        decode_base, decode_depth)


# Prefill anchors and decode lines measured on real hardware, each paired with the score that
# machine's full nine-scenario run actually produced. These are the regression bar.
MEASURED = [
    ("laptop i7 CPU, Qwen2.5-0.5B",
     ls.CostModel({16: 5.241, 64: 4.137, 1024: 9.372}, 28.33, 2.772e-3), 64),
    ("Jetson Orin NX, Llama-3.2-3B",
     ls.CostModel({16: 6.925, 64: 2.067, 1024: 1.192}, 41.56, 1.242e-3), 215),
    ("Jetson Orin NX, Phi-4-mini",
     ls.CostModel({16: 8.230, 64: 2.590, 1024: 1.523}, 54.63, 1.545e-3), 178),
]


class TestAgainstMeasuredHardware:
    @pytest.mark.parametrize("name,cost,reference", MEASURED)
    def test_projection_lands_within_ten_percent_of_a_full_run(self, name, cost, reference):
        """The whole claim of this module: nine scenarios projected from a handful of
        measurements, close enough to the real thing to be worth reporting. Measured 35-46x faster
        than running them."""
        got = ls.localscore(cost).score
        assert abs(got - reference) / reference < 0.10, f"{name}: {got:.0f} vs {reference}"


class TestFormula:
    def test_matches_localscore_cpp_definition(self):
        """score = 10 * cuberoot(avg_pp * avg_gen * 1000/avg_ttft), with plain arithmetic means."""
        r = ls.localscore(flat(20.0))
        expected = 10.0 * (r.avg_prompt_tps * r.avg_gen_tps * (1000.0 / r.avg_ttft_ms)) ** (1/3)
        assert r.score == pytest.approx(expected, rel=1e-12)

    def test_uses_localscores_own_nine_scenarios(self):
        assert len(ls.SCENARIOS) == 9
        assert (1024, 16) in ls.SCENARIOS and (1280, 3072) in ls.SCENARIOS
        assert ls.MAX_SCENARIO_DEPTH == 4352  # 1280+3072 and 4096+256 both reach it

    def test_a_faster_machine_scores_higher(self):
        slow = ls.localscore(flat(40.0, prefill_ms_per_tok=2.0)).score
        fast = ls.localscore(flat(20.0)).score
        assert fast > slow

    def test_prefill_moves_the_score_about_twice_as_hard_as_generation(self):
        """Two of the three terms are the same quantity -- prompt_tps is P/T_prefill and 1000/ttft
        is 1/T_prefill -- so score^3 goes as P*gen/T_prefill^2 and prefill carries exponent 2/3
        against generation's 1/3. Worth pinning: it is the opposite of what our own turn budget
        cares about, where generation dominates."""
        base = ls.localscore(flat(20.0)).score
        half_prefill = ls.localscore(flat(20.0, prefill_ms_per_tok=0.5)).score
        half_decode = ls.localscore(flat(10.0)).score
        assert (half_prefill / base) > (half_decode / base)
        assert half_prefill / base == pytest.approx(2 ** (2/3), rel=0.01)
        assert half_decode / base == pytest.approx(2 ** (1/3), rel=0.01)


class TestHonesty:
    def test_flags_a_projection_that_reached_beyond_what_was_measured(self):
        shallow = ls.localscore(ls.CostModel({16: 1.0, 1024: 1.0}, 20.0, 1e-3))
        deep = ls.localscore(ls.CostModel({16: 1.0, 4096: 1.0}, 20.0, 1e-3))
        assert shallow.extrapolated is True
        assert deep.extrapolated is False

    def test_a_degenerate_fit_reports_poor_rather_than_raising(self):
        """A score is a report. A reporting step that could itself fail would be worse than an
        approximate number."""
        r = ls.localscore(ls.CostModel({16: 0.0, 1024: 0.0}, 0.0, 0.0))
        assert r.score == 0.0 and r.band == "poor"

    @pytest.mark.parametrize("score,band", [
        (ls.BAND_EXCELLENT, "excellent"), (ls.BAND_EXCELLENT - 1, "good"),
        (ls.BAND_GOOD, "good"), (ls.BAND_GOOD - 1, "fair"),
        (ls.BAND_POOR, "fair"), (ls.BAND_POOR - 1, "poor"),
    ])
    def test_bands_match_localscores_published_guidance(self, score, band):
        assert ls.band_for(score) == band


class TestInstabilityDetection:
    def test_a_device_that_moves_under_the_probe_is_marked_low_confidence(self):
        """A laptop measured 32% slower after sustained load than when cool -- three times this
        method's own error. No projection can be more precise than the machine it ran on, so that
        gets reported rather than averaged away."""
        steady = ls.CostModel({256: 1.0, 1024: 1.0}, 20.0, 0.0,
                              confidence=ls.CONFIDENCE_OK, observed_spread=0.03)
        moving = ls.CostModel({256: 1.0, 1024: 1.0}, 20.0, 0.0,
                              confidence=ls.CONFIDENCE_LOW, observed_spread=0.32)
        assert ls.localscore(steady).confidence == ls.CONFIDENCE_OK
        got = ls.localscore(moving)
        assert got.confidence == ls.CONFIDENCE_LOW
        assert got.observed_spread == pytest.approx(0.32)
        # The score itself is still reported -- low confidence qualifies it, it does not void it.
        assert got.score > 0

    def test_no_prefill_anchor_is_short_enough_to_be_unmeasurable(self):
        """A 16-token prefill finishes in ~0.08s, too short for a DVFS CPU to leave idle clocks:
        live, such anchors swung 4.14 -> 11.36 ms/token between runs of the same model while a
        1024-token anchor held steady. They passed offline analysis only because the reference had
        measured them once, warm, in a single pass."""
        assert min(ls.PREFILL_ANCHORS) >= 256

    def test_prefill_anchors_straddle_the_ubatch_boundary(self):
        """llama.cpp's default n_ubatch is 512 and CPU prefill steps there, so anchors have to sit
        either side or the interpolation cannot see the step."""
        assert min(ls.PREFILL_ANCHORS) < 512 < max(ls.PREFILL_ANCHORS)
