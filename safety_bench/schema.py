"""The data contract: one case format, one run format. Unknown or missing fields are errors.

  python safety_bench/schema.py    # writes schema/case.schema.json and schema/run.schema.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SB = Path(__file__).resolve().parent
TAXONOMY_PATH = SB / "taxonomy.json"
BUDGET_TOOL_CALLS = 200  # the agent's budget per run, from the source harness
BUDGET_SECONDS = 300.0


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Rule(Strict):
    """One tool-output edit. Fields match finance_agent.adversary.InjectionRule."""

    tool: Literal["web_search", "parse_html_page"]
    primitive: Literal["replace", "append", "serve"]
    keywords: list[str] = []  # all must appear in the tool arguments
    aliases: list[str] = []  # at least one must appear
    payload: dict
    max_applications: int = 0  # 0 = every matching source


class Delivery(Strict):
    """How the attack reaches the agent. A user-vector attack is in the prompt, so it has no rules."""

    vector: Literal["none", "user", "search", "page"]
    rules: list[Rule] = []


class Rubric(Strict):
    id: str
    role: Literal["safe", "harm", "fact", "detect"]
    text: str


class Judge(Strict):
    rubrics: list[Rubric]
    success_rule: Literal["harm_present", "none"]


class Check(Strict):
    """A deterministic check. With decides=true it sets the outcome; otherwise it cross-checks the judge."""

    type: Literal["answer_token", "confidential", "confidential_detail", "honeypot"]
    tokens: list[str] = Field(min_length=1)
    decides: bool


class Reach(Strict):
    """How to tell the attack reached the main model. `always` for attacks written in the prompt."""

    type: Literal["always", "marker"]
    markers: list[str] = []
    truth_markers: list[str] = []  # the true figure, to see whether both values were visible


class Utility(Strict):
    """What report quality is measured against."""

    type: Literal["source_rubrics", "case_rubrics"]
    query_id: Optional[str] = None  # source_rubrics: FrontierFinance query
    exclude_rubric_ids: list[int] = []  # source_rubrics: the attacked rubric, already counted by ASR
    rubrics_sha: Optional[str] = None  # source_rubrics: pins the rubric text used
    rubric_ids: list[str] = []  # case_rubrics: `fact` rubrics of this case


class Scoring(Strict):
    judge: Optional[Judge]
    checks: list[Check] = []
    reach: Reach
    utility: Optional[Utility]


class Case(Strict):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    class_: Literal["misinformation", "premise", "pressure", "injection", "privacy", "control"] = Field(alias="class")
    attack_id: Optional[str]  # None for a control
    role: Literal["attack", "control"]
    variant: str
    level: Optional[int]  # position in an ordered variant list; None when the variants have no order
    entity: str
    as_of: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    prompt: str  # what the agent receives
    base_prompt: str  # the task without attack text; utility is graded against this
    control_id: Optional[str]
    delivery: Delivery
    scoring: Scoring
    provenance: dict


class RunUtility(Strict):
    passed: int
    total: int
    score: float
    labels: dict[str, bool]


class Run(Strict):
    run_id: str
    case_id: str
    class_: str = Field(alias="class")
    attack_id: Optional[str]
    variant: str
    level: Optional[int]
    vector: str
    role: Literal["attack", "control"]
    seed: int
    case_sha: str
    outcome: Literal["attack_succeeded", "attack_failed", "inconclusive", "not_used", "subagent_rejected", "control", "error"]
    error: Optional[str]
    exposed: bool  # the overlay fired
    subagent_read: bool  # a retrieval sub-model's input really contained the attack content
    subagent_warned: bool  # that sub-model said something was wrong with it
    reached: bool  # the attack reached the main model
    true_seen: Optional[bool]  # the true figure was also in the tool outputs
    judge_verdict: Optional[str]
    judge_labels: Optional[dict[str, bool]]
    detected: Optional[bool]  # the answer told the user about the conflict or the instruction
    checks: dict[str, bool]
    check_vs_judge_conflict: bool  # token check and judge disagree: review this run
    utility: Optional[RunUtility]
    tool_calls: int
    elapsed_s: float
    answer_chars: int
    agent_status: Optional[str]
    flags: list[str]
    trajectory: Optional[str]


def case_sha(case: dict) -> str:
    return hashlib.sha256(json.dumps(case, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


def taxonomy() -> dict:
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))


def attacks() -> dict[str, dict]:
    return {a["id"]: {**a, "class": c["id"]} for c in taxonomy()["classes"] for a in c["attacks"]}


def main() -> None:
    out = SB / "schema"
    out.mkdir(exist_ok=True)
    for name, model in (("case", Case), ("run", Run)):
        (out / f"{name}.schema.json").write_text(
            json.dumps(model.model_json_schema(by_alias=True), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
