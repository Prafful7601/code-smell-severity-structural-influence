"""
01_build_instances.py

Reads the raw MLCQ export and produces a clean instance-level table:
one row per (sample_id, smell) — 9,517 rows — with a consensus severity
label. This is the join key ('instance_id' = f"{sample_id}::{smell}")
used by both persistence.csv and complexity.csv.

Steps:
  1. Drop exact duplicate reviewer submissions (same reviewer_id, sample_id,
     smell, severity — i.e. accidental double-click resubmits). This drops
     2 rows (4 rows -> 2 distinct submissions) in the known dataset.
  2. Collapse repeated (reviewer_id, sample_id, smell) rows that differ only
     in review_timestamp are NOT expected after step 1 -- verified below.
  3. Compute consensus severity per (sample_id, smell) as the majority vote
     (mode) of severity across distinct reviewers. Ties (>1 severity level
     tied for the mode) are broken toward the HIGHER severity, per the
     project's own decision (documented in METHOD.md): understating severity
     would bias H2 toward the null, so ties resolve upward.
  4. is_from_industry_relevant_project is a repo-level flag with a stray
     "0,5" value in the raw export; preserved as-is (string) with a
     parsed numeric column added for convenience. Not otherwise used here.

Output: data/instances.csv, data/instances_dropped_duplicates.csv (audit trail)
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_XLSX = ROOT / "Test Smell Dataset.xlsx"
OUT_INSTANCES = ROOT / "data" / "instances.csv"
OUT_DUPES = ROOT / "data" / "instances_dropped_duplicates.csv"

SEVERITY_ORDER = ["none", "minor", "major", "critical"]
SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITY_ORDER)}


def consensus_severity(sev_series: pd.Series) -> tuple[str, int, bool]:
    """Majority-vote severity across distinct reviewer rows for one instance.

    Returns (consensus_severity, n_reviewers, was_tied).
    Ties broken toward the higher severity (see module docstring).
    """
    counts = sev_series.value_counts()
    top = counts.max()
    tied_values = counts[counts == top].index.tolist()
    was_tied = len(tied_values) > 1
    # break ties toward higher severity
    chosen = max(tied_values, key=lambda s: SEVERITY_RANK[s])
    return chosen, len(sev_series), was_tied


def main():
    df = pd.read_excel(RAW_XLSX)
    assert len(df) == 14739, f"expected 14739 rows, got {len(df)}"

    # --- Step 1: drop exact duplicate reviewer submissions ---
    key_cols = ["reviewer_id", "sample_id", "smell", "severity"]
    dupe_mask = df.duplicated(subset=key_cols, keep="first")
    dropped = df[dupe_mask].copy()
    dropped.to_csv(OUT_DUPES, index=False)
    df = df[~dupe_mask].copy()
    print(f"Dropped {dupe_mask.sum()} exact duplicate reviewer submissions "
          f"-> {len(df)} rows remain.")

    # sanity: no remaining duplicate (reviewer_id, sample_id, smell) with
    # *different* severity should silently collide -- if this fires, a
    # reviewer genuinely resubmitted a different rating and we need a
    # different rule (currently: none observed in the known dataset).
    remaining_dupe_keys = df.duplicated(subset=["reviewer_id", "sample_id", "smell"], keep=False)
    if remaining_dupe_keys.any():
        print("WARNING: reviewer resubmissions with DIFFERING severity found:")
        print(df[remaining_dupe_keys].sort_values(["sample_id", "smell"]))

    # --- Step 2/3: consensus severity per (sample_id, smell) ---
    rows = []
    for (sample_id, smell), g in df.groupby(["sample_id", "smell"], sort=False):
        sev, n_rev, tied = consensus_severity(g["severity"])
        first = g.iloc[0]
        rows.append({
            "instance_id": f"{sample_id}::{smell}",
            "sample_id": sample_id,
            "smell": smell,
            "type": first["type"],
            "code_name": first["code_name"],
            "repository": first["repository"],
            "commit_hash": first["commit_hash"],
            "path": first["path"],
            "start_line": first["start_line"],
            "end_line": first["end_line"],
            "link": first["link"],
            "is_from_industry_relevant_project_raw": first["is_from_industry_relevant_project"],
            "n_reviewers": n_rev,
            "severity_consensus": sev,
            "severity_tied": tied,
            "severity_values_seen": ",".join(sorted(g["severity"].unique(), key=lambda s: SEVERITY_RANK[s])),
        })

    inst = pd.DataFrame(rows)
    assert len(inst) == 9517, f"expected 9517 instances, got {len(inst)}"

    # sanity check against the user's stated distribution (90.4/5.4/3.4/0.8)
    dist = (inst["severity_consensus"].value_counts(normalize=True) * 100).round(2)
    print("\nConsensus severity distribution (%):")
    print(dist)

    inst = inst.sort_values(["sample_id", "smell"]).reset_index(drop=True)
    inst.to_csv(OUT_INSTANCES, index=False)
    print(f"\nWrote {len(inst)} instances -> {OUT_INSTANCES}")
    print(f"Tied instances (broken toward higher severity): {inst['severity_tied'].sum()}")


if __name__ == "__main__":
    main()
