"""Where the benchmark finds its data.

The public repository ships only cases that contain no FrontierFinance text. The FrontierFinance dataset is gated
on Hugging Face, and its card asks users not to reshare it outside a gated or private repository, so the cases
built from it (false-figure attacks M1-M3, hidden-instruction attacks I1-I4, and their baselines) are rebuilt
locally from a copy the user downloads:

  FRONTIER_FINANCE_DATA   path to frontier_finance_public.jsonl (default: data/frontier_finance_public.jsonl)

  dataset/cases.jsonl       public cases, in the repository
  dataset/cases_full.jsonl  all cases, built locally by build/build_dataset.py when the source data is present
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SB = ROOT / "safety_bench"
DATASET_DIR = SB / "dataset"
PUBLIC_CASES = DATASET_DIR / "cases.jsonl"
FULL_CASES = DATASET_DIR / "cases_full.jsonl"
SOURCE_FILE_NAME = "frontier_finance_public.jsonl"
SOURCE_URL = "https://huggingface.co/datasets/samaya-ai/FrontierFinance"
# The file the cases were frozen with (Hugging Face upload 0297674d, 4,846,022 bytes). A different file would
# rebuild different cases, so the build refuses it instead of producing them silently.
SOURCE_SHA256 = "a82874d7a587baf6f1ebe79b95fa1c3090260d3661c544f4496056d338e313c4"


def source_data_path() -> Path:
    configured = os.getenv("FRONTIER_FINANCE_DATA", "").strip()
    return Path(configured) if configured else ROOT / "data" / SOURCE_FILE_NAME


def has_source_data() -> bool:
    return source_data_path().is_file()


def source_data_sha256() -> str:
    import hashlib

    return hashlib.sha256(source_data_path().read_bytes()).hexdigest()


def check_source_version() -> None:
    """Raise if the local FrontierFinance file is not the one the cases were frozen with."""
    got = source_data_sha256()
    if got != SOURCE_SHA256:
        raise SystemExit(f"{source_data_path()} has SHA-256 {got}, but the cases were frozen with {SOURCE_SHA256}. "
                         f"Download the frozen version from {SOURCE_URL} (file frontier_finance_public.jsonl, upload 0297674d).")


def missing_source_message() -> str:
    return (f"FrontierFinance data not found at {source_data_path()}. Download {SOURCE_FILE_NAME} from {SOURCE_URL} "
            "(access is gated), then set FRONTIER_FINANCE_DATA or place it in data/.")


def cases_path() -> Path:
    """The full case set when it has been built locally, otherwise the public one."""
    return FULL_CASES if FULL_CASES.exists() else PUBLIC_CASES
