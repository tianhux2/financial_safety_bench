"""Run a batch: preflight, manifest, parallel workers, merged results.

Parallelism has to be process-level: MAX_END_DATE is a module global, and each process needs its own runs
dir because a save rewrites <runs-dir>/index.json. A batch is resumable: rerun the same command and finished
runs are skipped, but a dataset change under an existing tag is refused, so one tag is always one dataset.

Stages: stage 1 is one seed of everything. Stage 2 adds a second seed of the attacks marked stage_2 in
taxonomy.json and all controls, so that each ordered level has at least 16 runs.

  python safety_bench/run.py --tag smoke --ids M1_aig-r9_assertive,CTL_aig-r9 --workers 2
  python safety_bench/run.py --tag main --preview 3 --workers 10   # a rule-based sample of stage 1
  python safety_bench/run.py --tag main --stage 1 --workers 6      # finished runs are skipped
  python safety_bench/run.py --tag main --stage 2 --workers 6
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

SB = Path(__file__).resolve().parent
REPO = SB.parent
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env", override=True)

from safety_bench.judge import DEFAULT_JUDGE_MODELS, PROMPT_VERSION, Judge, ff_revision  # noqa: E402
from safety_bench.paths import cases_path  # noqa: E402
from safety_bench.schema import TAXONOMY_PATH, attacks  # noqa: E402

DATASET = cases_path()
RESULTS = SB / "results"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def preflight() -> None:
    """Fail in seconds, not after a batch of identical errors."""
    try:
        reply = Judge.build(DEFAULT_JUDGE_MODELS[0], max_tokens=1024, timeout=60.0, max_retries=0).complete("Reply with OK.", "ping")
    except Exception as e:
        raise SystemExit(f"preflight failed, nothing was run: {str(e)[:300]}")
    print(f"preflight: {DEFAULT_JUDGE_MODELS[0]} answered {str(reply)[:20]!r}", flush=True)
    from finance_agent.tools import TavilyWebSearch

    result = asyncio.run(TavilyWebSearch().execute({"search_query": "Apple 2024 annual report"}, {}, logging.getLogger("preflight")))
    if getattr(result, "error", None) or "usage limit" in str(getattr(result, "output", result)):
        raise SystemExit(f"preflight failed, nothing was run: web search: {str(getattr(result, 'output', result))[:200]}")
    print("preflight: web search answered", flush=True)
    import os

    if not os.getenv("SEC_USER_AGENT", "").strip():
        raise SystemExit("preflight failed, nothing was run: set SEC_USER_AGENT to your name and email; SEC EDGAR requires a contact")
    print(f"preflight: {DATASET.name} ({sum(1 for l in DATASET.read_text(encoding='utf-8').splitlines() if l.strip())} cases)", flush=True)


def manifest(tag: str) -> dict:
    return {"tag": tag, "dataset_sha": sha(DATASET), "taxonomy_sha": sha(TAXONOMY_PATH), "git_commit": git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")), "agent_model": "gpt-6-luna", "judge_models": list(DEFAULT_JUDGE_MODELS),
            "judge_prompt": PROMPT_VERSION, "frontier_finance": ff_revision()}


def write_or_check_manifest(tag_dir: Path, tag: str) -> None:
    path = tag_dir / "manifest.json"
    current = manifest(tag)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["dataset_sha"] != current["dataset_sha"]:
            raise SystemExit(f"tag {tag} was started on dataset {old['dataset_sha']}, now {current['dataset_sha']}: use a new tag")
        return
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    if current["git_dirty"]:
        print("warning: the working tree has uncommitted changes, so this batch is not pinned to a commit", flush=True)


PREVIEW_INJECTION_ENTITY = "AIG"
PREVIEW_INJECTION_STYLES = ("naive", "task_aligned", "retrieval_directed")


def preview_ids(cases: list[dict], per_variant: int) -> list[str]:
    """A sample of stage 1 by a fixed rule, in dataset order, plus the controls of what it picked.

    The first N cases of each variant. Injection has four attacks of 44 cells each, so it keeps one company and
    three styles: the plainest, the one without an imperative, and the one addressed to the retrieval sub-model.
    """
    picked, seen = [], collections.Counter()
    for c in cases:
        if c["role"] != "attack":
            continue
        if c["class"] == "injection":
            if c["entity"] == PREVIEW_INJECTION_ENTITY and c["variant"] in PREVIEW_INJECTION_STYLES:
                picked.append(c)
            continue
        key = (c["attack_id"], c["variant"])
        if seen[key] < per_variant:
            seen[key] += 1
            picked.append(c)
    controls = list(dict.fromkeys(c["control_id"] for c in picked if c["control_id"]))
    return controls + [c["id"] for c in picked]


def plan(args: argparse.Namespace, cases: list[dict]) -> list[tuple[str, int]]:
    if args.preview:
        return [(i, 0) for i in preview_ids(cases, args.preview)]
    if args.ids:
        want = args.ids.split(",")
        missing = set(want) - {c["id"] for c in cases}
        if missing:
            raise SystemExit(f"unknown ids: {sorted(missing)}")
        return [(i, 0) for i in want]
    if args.stage == 1:
        return [(c["id"], 0) for c in cases]
    stage_2 = {k for k, a in attacks().items() if a["stage_2"]}
    return [(c["id"], 1) for c in cases if c["role"] == "control" or c["attack_id"] in stage_2]


def done_runs(tag_dir: Path) -> set[str]:
    """Finished runs. An error row does not count, so a resume retries it."""
    done = set()
    for p in tag_dir.glob("*.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row["outcome"] != "error":
                    done.add(row["run_id"])
    return done


def merge(tag_dir: Path) -> int:
    """Worker files -> runs.jsonl, keeping the last row of each run."""
    rows: dict[str, dict] = {}
    paths = ([tag_dir / "runs.jsonl"] if (tag_dir / "runs.jsonl").exists() else []) + sorted(tag_dir.glob("_p*.jsonl"))
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["run_id"]] = row
    (tag_dir / "runs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows.values()), encoding="utf-8")
    for p in tag_dir.glob("_p*.jsonl"):
        p.unlink()
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ids", help="comma-separated case ids (seed 0)")
    ap.add_argument("--preview", type=int, help="first N cases of each variant, with their controls")
    ap.add_argument("--stage", type=int, choices=[1, 2], default=1)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cases = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    tag_dir = RESULTS / args.tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    preflight()
    write_or_check_manifest(tag_dir, args.tag)
    merge(tag_dir)  # picks up rows left by an interrupted batch
    finished = done_runs(tag_dir)
    work = [(i, s) for i, s in plan(args, cases) if f"{i}__s{s}" not in finished]
    print(f"{args.tag}: {len(work)} runs to do, {len(finished)} finished, {args.workers} workers", flush=True)

    procs = []
    for i in range(args.workers):
        shard = work[i::args.workers]
        if not shard:
            continue
        ids = tag_dir / f"_ids_{i}.json"
        ids.write_text(json.dumps(shard), encoding="utf-8")
        log = (tag_dir / f"_worker_{i}.log").open("a", encoding="utf-8")
        procs.append(subprocess.Popen(
            [sys.executable, str(SB / "evaluate.py"), "--ids-file", str(ids), "--out", str(tag_dir / f"_p{i}.jsonl"),
             "--runs-dir", f"runs/{args.tag}/p{i}"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    n = merge(tag_dir)
    for p in tag_dir.glob("_ids_*.json"):
        p.unlink()
    print(f"worker exit codes {codes}; {n} runs in {tag_dir / 'runs.jsonl'}", flush=True)
    if any(codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
