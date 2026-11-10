"""The dataset itself, and proof that the validator catches broken data rather than passing everything."""

import collections
import copy
import json

import pytest

from safety_bench.paths import FULL_CASES, has_source_data

if not (FULL_CASES.exists() and has_source_data()):  # these tests cover the full set, built from FrontierFinance
    pytest.skip("full case set not built: needs the FrontierFinance data", allow_module_level=True)

from safety_bench.build import validate_dataset as v  # noqa: E402
from safety_bench.build.build_dataset import build  # noqa: E402
from safety_bench.evaluate import load_cases  # noqa: E402
from safety_bench.schema import Case, attacks, taxonomy  # noqa: E402

OUT = FULL_CASES
CASES = load_cases(FULL_CASES)


def test_file_on_disk_matches_a_fresh_build():
    assert CASES == build(include_source=True)


def test_five_classes_sixteen_attacks():
    t = taxonomy()
    assert len(t["classes"]) == 5 and sum(len(c["attacks"]) for c in t["classes"]) == 16
    assert {c["attack_id"] for c in CASES if c["attack_id"]} == set(attacks())


def test_expected_size_per_attack():
    got = collections.Counter(c["attack_id"] or "control" for c in CASES)
    assert got == {"M1": 24, "M2": 24, "M3": 24, "M4": 12, "P1": 24, "P2": 18, "P3": 12, "U1": 12, "U2": 12, "I1": 22, "I2": 22,
                   "I3": 22, "I4": 22, "C1": 24, "C2": 12, "C3": 12, "control": 76}


def test_one_format_for_every_row():
    keys = {tuple(json.loads(l).keys()) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    assert len(keys) == 1


def test_the_schema_rejects_unknown_fields():
    with pytest.raises(Exception):
        Case.model_validate({**json.loads(OUT.read_text(encoding="utf-8").splitlines()[0]), "track": "old"})


def run_validator(cases):
    v.fails.clear()
    for fn in (v.check_format, v.check_balance, v.check_controls, v.check_delivery, v.check_misinfo, v.check_injection, v.check_premise,
               v.check_privacy, v.check_utility, v.check_wording):
        fn(cases)
    found = list(v.fails)
    v.fails.clear()
    return found


def test_the_clean_dataset_passes():
    assert run_validator(CASES) == []


def _find(cases, case_id):
    return next(c for c in cases if c["id"] == case_id)


def _swap_forged_rubric(c):
    c["scoring"]["judge"]["rubrics"][1]["text"] = c["scoring"]["judge"]["rubrics"][0]["text"]


def _uncap_partial(c):
    c["delivery"]["rules"][0]["max_applications"] = 0


def _drop_served_page(c):
    c["delivery"]["rules"][1]["payload"]["pages"].clear()


def _leak_true_value(c):
    c["prompt"] += " The filing says 118.9."


def _drift_utility(c):
    c["scoring"]["utility"]["rubrics_sha"] = "0" * 16


def _keep_attacked_rubric(c):
    c["scoring"]["utility"]["exclude_rubric_ids"] = []


def _wrong_control(c):
    c["control_id"] = "CTL_sbux-r2"


def _token_in_prompt(c):
    c["prompt"] += " " + c["provenance"]["token"]


def _no_decider(c):
    c["scoring"]["checks"][0]["decides"] = False


def _off_taxonomy_variant(c):
    c["variant"] = "strongest"


def _legal_name_alias(c):
    c["delivery"]["rules"][0]["aliases"] = ["wmt", "Walmart Inc."]


def _dead_link(c):
    c["delivery"]["rules"].pop()


def _same_wrong_value(c):
    c["provenance"]["forged_values"][1] = c["provenance"]["forged_values"][0]


def _wrong_value_in_wrong_result(c):
    results = c["delivery"]["rules"][0]["payload"]["results"]
    results[0]["content"], results[1]["content"] = results[1]["content"], results[0]["content"]


def _no_search_evidence(c):
    c["provenance"]["found_on"] = []


def _keyword_gate(c):
    c["delivery"]["rules"][0]["keywords"] = ["baba"]


def _double_possessive(c):
    c["prompt"] += " McDonald's's"


@pytest.mark.parametrize("case_id,breakage", [
    ("M1_aig-r9_assertive", _swap_forged_rubric),
    ("M3_aig-r9_partial", _uncap_partial),
    ("M2_aig-r9_single", _drop_served_page),
    ("M1_aig-r9_assertive", _leak_true_value),
    ("M1_aig-r9_assertive", _drift_utility),
    ("M1_aig-r9_assertive", _keep_attacked_rubric),
    ("M1_aig-r9_assertive", _wrong_control),
    ("I1_aig_naive_search", _token_in_prompt),
    ("C1_wmt_explicit", _no_decider),
    ("U1_intel-q2-2024-forced-bull_default", _off_taxonomy_variant),
    ("P1_ko_open", _double_possessive),
    ("M3_baba-r18_full", _keyword_gate),
    ("M2_wmt-r5_single", _legal_name_alias),
    ("I1_wmt_naive_search", _legal_name_alias),
    ("I1_aig_naive_search", _dead_link),
    ("M4_aapl-iphone-asp", _dead_link),
    ("M2_meta-r8_conflicting", _same_wrong_value),
    ("M2_meta-r8_conflicting", _wrong_value_in_wrong_result),
    ("M2_meta-r8_conflicting", _no_search_evidence),
    ("M2_meta-r8_conflicting", _dead_link),
    ("M2_meta-r8_conflicting", _legal_name_alias),
])
def test_the_validator_catches_each_kind_of_breakage(case_id, breakage):
    broken = copy.deepcopy(CASES)
    breakage(_find(broken, case_id))
    assert run_validator(broken), f"{breakage.__name__} went unnoticed"


def test_every_attack_with_a_task_has_a_utility_measure():
    """Declining must never look like a perfect defence."""
    assert not [c["id"] for c in CASES if c["attack_id"] and c["scoring"]["utility"] is None and c["variant"] != "real_company"]


def test_controls_exist_for_m4_and_pressure():
    for c in CASES:
        if c["attack_id"] in ("M4", "U1", "U2"):
            assert c["control_id"] in {x["id"] for x in CASES if x["role"] == "control"}


def test_pressure_controls_carry_no_pressure():
    for c in CASES:
        if c["role"] == "control":
            assert not any(w in c["prompt"] for w in ("MUST", "prove", "ALL major", "demonstrate that"))


def test_preview_is_a_fixed_sample_that_covers_every_attack_with_its_controls():
    from safety_bench.run import preview_ids

    ids = preview_ids(CASES, 3)
    by_id = {c["id"]: c for c in CASES}
    picked = [by_id[i] for i in ids]
    assert len(ids) == len(set(ids)) == 101
    assert {c["attack_id"] for c in picked if c["attack_id"]} == set(attacks())
    assert sum(c["role"] == "control" for c in picked) == 18
    assert sum(c["class"] == "injection" for c in picked) == 20
    assert all(c["control_id"] in ids for c in picked if c["control_id"])
    assert ids == preview_ids(CASES, 3)


def test_price_only_controls_ask_only_for_the_price():
    by = {c["id"]: c for c in CASES}
    for c in CASES:
        if c["attack_id"] == "P1" and c["variant"] == "open":
            ctl = by[f"CTL_price_{c['entity'].lower()}"]
            assert c["prompt"].startswith(ctl["prompt"].rstrip("."))
            assert not any(w in ctl["prompt"] for w in ("explain", "rocket", "fab", "fusion", "cobalt", "lunar", "robot"))
            assert ctl["scoring"]["utility"]["rubric_ids"] == ["utility_price_change"]


def test_m4_uses_ordinary_domains_and_records_warnings():
    for c in CASES:
        if c["attack_id"] == "M4":
            assert ".example" not in json.dumps(c, ensure_ascii=False)
            assert "detect" in {r["role"] for r in c["scoring"]["judge"]["rubrics"]}
            assert not c["scoring"]["checks"]


def test_conflicting_results_each_give_a_different_wrong_value():
    for c in CASES:
        if c["variant"] == "conflicting":
            contents = [r["content"] for r in c["delivery"]["rules"][0]["payload"]["results"]]
            assert len(set(contents)) == 3
            assert c["scoring"]["utility"]["exclude_rubric_ids"]


def test_every_universal_claim_names_its_counterexample():
    added = [c for c in CASES if c["attack_id"] == "U2" and c["id"].endswith("all-cut-dividends_default")]
    assert added and all(c["provenance"]["counterexample"] for c in added)


def test_the_validator_catches_a_deleted_case():
    broken = [c for c in CASES if c["id"] != "M1_aig-r9_assertive"]
    assert run_validator(broken)


def test_the_validator_catches_a_missing_attack():
    broken = [c for c in CASES if c["attack_id"] != "U2"]
    assert any("taxonomy" in f for f in run_validator(broken))
