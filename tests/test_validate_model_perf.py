"""scripts/validate_model_perf.py -- and specifically, proof that it can go red.

CLAUDE.md's "Tests and gates" rule 2: a check that cannot fail is not a check, and it has to fail
for THE FAILURE IT NAMES. This validator exists to catch model_perf's grade promising a turn that
really times out, so the test that matters is the one where a model's real cost curve diverges from
the fit by more than the safety factor absorbs, and the validator notices.

The stub router's `decode_quadratic` knob is that lever: it is invisible at the two prompt lengths
the probe measures and dominant out where N_fit lands -- exactly the shape of the mistake this gate
is supposed to catch, and exactly the one a two-point linear fit cannot see for itself.
"""

from __future__ import annotations

import re

import validate_model_perf as vmp

# Per-token costs chosen so the stub's real sleeps keep a whole run near a second, while preserving
# the SHAPE each assertion is about. LINEAR is what the fit assumes; SUPERLINEAR adds a quadratic
# term worth almost nothing at the depths the probe samples and more than everything else out at
# N_fit.
LINEAR = dict(prefill_base=0.002, prefill_slope=2e-8, decode_base=0.1, decode_slope=5e-6)
# Sized so the divergence out at N_fit exceeds the safety factor -- a smaller one is correctly
# ABSORBED by that factor and passes, which is the behaviour the factor exists for.
SUPERLINEAR = dict(LINEAR, decode_quadratic=6e-10)

TABLE_ROW = re.compile(r"^\s*\d+\s+[\d,]+\s+[\d.]+s\s+[\d.]+s")


# Every measured turn is sized to fill the budget, so a real 155s one would make this suite take
# minutes per case. A small budget keeps the SHAPE identical -- claim, measure, compare -- at a
# scale a test can afford.
TEST_BUDGET_MS = 400.0


def _run(router, ctx_size=40960, budget_ms=TEST_BUDGET_MS, **kwargs) -> int:
    argv = ["--model", "org/repo:Q4_K_M", "--base-url", router.base_url, "--warm",
            "--ctx-size", str(ctx_size), "--budget-ms", str(budget_ms)]
    for key, value in kwargs.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return vmp.main(argv)


class TestValidatorGate:
    def test_passes_when_the_grades_promise_holds(self, router_stub, capsys):
        """A model whose real cost IS linear in depth -- the assumption the fit makes -- must come
        back clean, or the gate is useless noise that nobody will keep running."""
        with router_stub(**LINEAR) as router:
            rc = _run(router)

        out = capsys.readouterr().out
        assert rc == 0, out
        assert "PASS" in out

    def test_fails_when_the_real_cost_outruns_what_the_safety_factor_absorbs(self, router_stub, capsys):
        """The failure this gate is named after: a superlinear decode cost the two-point probe
        cannot see, so the grade promises an N_fit that does not actually fit."""
        with router_stub(**SUPERLINEAR) as router:
            rc = _run(router)

        out = capsys.readouterr().out
        assert rc == 1, out
        assert "FAIL" in out
        assert "under-predicted" in out or "times out" in out

    def test_measures_a_turn_of_exactly_the_length_the_model_claims(self, router_stub, capsys):
        """The claim is "this many output tokens arrive in time", so the turn measured has to be
        that long -- a shorter one would validate something nobody promised."""
        with router_stub(**LINEAR) as router:
            _run(router)
        out = capsys.readouterr().out
        assert "claimed out" in out
        assert "ok" in out

    def test_reports_whether_the_safety_factor_covered_the_error(self, router_stub, capsys):
        with router_stub(**LINEAR) as router:
            _run(router)
        out = capsys.readouterr().out
        assert "worst under-prediction" in out
        assert "absorbed by safety factor" in out

    def test_a_refused_model_has_no_promise_to_check(self, router_stub, capsys):
        with router_stub(decode_base=900.0) as router:
            rc = _run(router)
        out = capsys.readouterr().out
        assert rc == 0
        assert "Nothing to validate" in out

    def test_max_input_limits_the_depths_tested(self, router_stub, capsys):
        with router_stub(**LINEAR) as router:
            _run(router, max_input=4096)
        rows = [l for l in capsys.readouterr().out.splitlines() if TABLE_ROW.match(l)]
        assert 0 < len(rows) <= 2  # depths 1024 and 4096 only
