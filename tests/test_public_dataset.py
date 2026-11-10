"""The public case file: what anyone can run without the gated FrontierFinance data."""

import collections
import json

from safety_bench.build import validate_dataset as v
from safety_bench.build.build_dataset import build, is_source_derived
from safety_bench.evaluate import load_cases, score_run
from safety_bench.judge import RubricJudge
from safety_bench.paths import PUBLIC_CASES
from tests.test_judge import StubJudge

CASES = load_cases(PUBLIC_CASES)


def test_public_file_matches_a_build_without_source_data():
    assert CASES == build(include_source=False)


def test_public_file_has_no_source_derived_case():
    assert not any(is_source_derived(c) for c in CASES)
    assert not {c["attack_id"] for c in CASES} & v.SOURCE_ONLY_ATTACKS


def test_expected_public_size():
    got = collections.Counter(c["attack_id"] or "control" for c in CASES)
    assert got == {"M4": 12, "P1": 24, "P2": 18, "P3": 12, "U1": 12, "U2": 12, "C1": 24, "C2": 12, "C3": 12, "control": 60}


def test_one_format_for_every_row():
    keys = {tuple(json.loads(l).keys()) for l in PUBLIC_CASES.read_text(encoding="utf-8").splitlines() if l.strip()}
    assert len(keys) == 1


def test_the_public_set_passes_the_offline_checks():
    v.fails.clear()
    for fn in (v.check_format, v.check_balance, v.check_controls, v.check_delivery, v.check_misinfo, v.check_injection,
               v.check_premise, v.check_privacy, v.check_utility, v.check_wording):
        fn(CASES)
    found = list(v.fails)
    v.fails.clear()
    assert found == []


def test_a_leak_is_scored_without_any_source_data():
    case = next(c for c in CASES if c["id"] == "C3_wmt_default")
    run = {"steps": [{"tool_name": "web_search", "args": {"search_query": "Walmart Project Bluefinch"}, "output": ""}]}

    class Judge(StubJudge):
        def complete(self, system, user):
            return json.dumps({"0": {"reason": "r", "label": True}})

    out = score_run(case, "report", run, True, RubricJudge([Judge()]))
    assert out["outcome"] == "attack_succeeded"


def test_a_different_source_file_is_refused(tmp_path, monkeypatch):
    """A changed FrontierFinance file must stop the build, not rebuild different cases silently."""
    import pytest

    from safety_bench.build.build_dataset import source_rubrics

    other = tmp_path / "frontier_finance_public.jsonl"
    other.write_text('{"query_id": "x", "query": "q", "query_date": "2025-01-01", "rubrics": []}\n', encoding="utf-8")
    monkeypatch.setenv("FRONTIER_FINANCE_DATA", str(other))
    with pytest.raises(SystemExit, match="frozen"):
        source_rubrics()
