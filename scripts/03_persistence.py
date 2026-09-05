"""
03_persistence.py

For each (sample_id, smell) instance, walk the file's history FORWARD from
its labelled commit and determine how many subsequent revisions the smell
survived, using the cheap entity-alive + size/shape proxy documented in
scripts/lib/entity_finder.py and METHOD.md.

Operationalization (agreed with the user before running):
  - "Revision" = a commit that touches the entity's file, found via
    `git log --follow --reverse` on the labelled path.
  - Persistence (revisions) = number of subsequent touching-revisions,
    starting the count at the labelled commit (revision 0), up to and
    NOT including the first revision where the entity is gone or its
    smell-condition proxy no longer holds.
  - Persistence (days) = calendar days between the labelled commit's
    author date and the "death" revision's author date (or HEAD's author
    date if right-censored).
  - censored = True if the entity is still alive with the condition still
    holding at the last observed revision (i.e. the file's history, as
    seen by --follow, runs out without the smell disappearing).

Resumable: writes one row per instance to data/persistence.csv as it goes,
and skips instance_ids already present in that file on a re-run. Failures
(repo not cloned, commit unreachable, path never found, parse errors)
are appended to failures.csv with a reason -- these instances get no
persistence.csv row.

Usage:
  python scripts/03_persistence.py --repo-list pilot/pilot_repos.txt
  python scripts/03_persistence.py --all
"""
import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.entity_finder import find_class_entity, find_method_entity, SMELL_CONDITION_FN
from lib.git_utils import run_git, resolve_head_sha, is_ancestor, find_remapped_commit, read_blob

ROOT = Path(__file__).resolve().parent.parent
REPOS_DIR = ROOT / "cache" / "repos"
INSTANCES_CSV = ROOT / "data" / "instances.csv"
OUT_CSV = ROOT / "data" / "persistence.csv"
FAILURES_CSV = ROOT / "failures.csv"

OUT_FIELDS = [
    "instance_id", "sample_id", "smell", "severity_consensus", "repository",
    "path_at_label", "code_name", "original_commit_hash", "commit_remapped",
    "effective_commit_hash", "head_sha_used", "n_revisions_observed", "persistence_revisions",
    "persistence_days", "days_anomalous_negative", "censored", "death_reason",
    "labelled_commit_date", "death_commit", "death_commit_date",
    "last_observed_commit", "last_observed_commit_date", "proxy_method",
]

FAILURE_FIELDS = ["repository", "instance_id", "sample_id", "smell", "stage", "reason", "detail", "timestamp"]


def append_failure(repository, instance_id, sample_id, smell, stage, reason, detail):
    exists = FAILURES_CSV.exists()
    with open(FAILURES_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FAILURE_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({
            "repository": repository, "instance_id": instance_id, "sample_id": sample_id,
            "smell": smell, "stage": stage, "reason": reason, "detail": str(detail)[:500],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })


def local_dir_for(repo_url: str) -> Path:
    owner_repo = repo_url.split(":", 1)[1].removesuffix(".git")
    return REPOS_DIR / owner_repo.replace("/", "__")


def commit_committer_date(repo_dir: Path, commit: str) -> str:
    """Committer date (%cd), NOT author date (%ad) -- author date reflects
    when a patch was originally written, which for a branch merged later
    can be EARLIER than an ancestor's author date (a PR opened before, but
    merged after, another commit). Committer date tracks when the commit
    actually landed on the branch we're walking, so it's monotonic along
    the commit_hash..HEAD range in the way "calendar days elapsed" needs.
    Verified against real data: see METHOD.md for a concrete example where
    author date alone produced a negative day count."""
    proc = run_git(repo_dir, ["log", "-1", "--format=%cd", "--date=iso-strict", commit])
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return proc.stdout.strip()


def file_history_after(repo_dir: Path, path: str, labelled_commit: str, head: str):
    """History of `path` STRICTLY AFTER labelled_commit, oldest-first, as
    (commit_hash, committer_date_iso, path_at_that_commit). labelled_commit
    is a repo-wide snapshot commit that need not itself have touched `path`
    (it's the review timestamp's commit, not a commit in this file's own
    log) -- so we can't look up labelled_commit inside `git log -- path`.
    Instead we walk the commit RANGE labelled_commit..HEAD with --follow,
    which is git's documented way to get "commits reachable from HEAD but
    not from labelled_commit" that touched path, with rename-tracking
    best-effort across the range boundary (a known git rough edge -- see
    METHOD.md). Uses committer date (%cd), not author date -- see
    commit_committer_date() docstring."""
    proc = run_git(repo_dir, [
        "log", "--follow", "--reverse", "--format=COMMIT|%H|%cd", "--date=iso-strict",
        "--name-only", f"{labelled_commit}..{head}", "--", path,
    ], timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())

    entries = []
    cur_hash, cur_date = None, None
    for line in proc.stdout.splitlines():
        if line.startswith("COMMIT|"):
            _, h, d = line.split("|", 2)
            cur_hash, cur_date = h, d
        elif line.strip():
            entries.append((cur_hash, cur_date, line.strip()))
    return entries


def entity_status(source: str, smell: str, code_name: str):
    """Returns (alive: bool, condition_holds: bool|None)."""
    is_class_smell = smell in ("blob", "data class")
    simple = code_name.split(".")[-1].split("#")[-1].split(" ")[0]
    if is_class_smell:
        simple_class = code_name.split("#")[0].split(".")[-1]
        span = find_class_entity(source, simple_class)
        if span is None:
            return False, None
        holds = SMELL_CONDITION_FN[smell](span)
        return True, holds
    else:
        # function code_name: FQCN(.Inner)*#methodName p1|p2|... OR
        # FQCN.methodName ... (constructor, methodName == last class segment)
        if "#" in code_name:
            head, rest = code_name.split("#", 1)
            parts = rest.split(" ", 1)
            method_name = parts[0]
            params = parts[1] if len(parts) > 1 else ""
        else:
            # split on the FIRST space, not the last: the params blob after
            # it can itself contain spaces (generic bounds like
            # "FunctionExpression<Function2<T, Integer, Enumerable<TResult>>>"
            # have a space after every comma), so rsplit(" ", 1) would grab
            # a split point inside the params instead of after method_name.
            # Confirmed on real data: this previously mis-parsed
            # apache/calcite's "QueryableDefaults.selectManyN" instance.
            head_dotted = code_name.split(" ", 1)[0]
            head = ".".join(head_dotted.split(".")[:-1])
            method_name = head_dotted.split(".")[-1]
            params = code_name.split(" ", 1)[1] if " " in code_name else ""
        n_params = len([p for p in params.split("|") if p]) if params else 0
        span = find_method_entity(source, method_name, n_params)
        if span is None:
            return False, None
        holds = SMELL_CONDITION_FN[smell](span)
        return True, holds


def process_instance(row, repo_dir: Path, effective_commit: str, commit_remapped: bool, head_sha: str):
    instance_id = row["instance_id"]
    smell = row["smell"]
    original_commit = row["commit_hash"]
    labelled_commit = effective_commit  # what we actually walk from -- may be a remapped hash
    path = row["path"].lstrip("/")
    code_name = row["code_name"]

    try:
        label_date = commit_committer_date(repo_dir, labelled_commit)
        forward = file_history_after(repo_dir, path, labelled_commit, head_sha)
    except Exception as e:
        append_failure(row["repository"], instance_id, row["sample_id"], smell,
                        "persistence_history", "git_log_failed", e)
        return None

    # baseline (revision 0) is the labelled commit itself -- verify the
    # entity is actually present there before walking forward; if not, the
    # labelled commit's own content doesn't contain the reviewed entity
    # (e.g. code_name/path mismatch) and we can't establish a starting point.
    baseline_source = read_blob(repo_dir, labelled_commit, path)
    if baseline_source is None:
        append_failure(row["repository"], instance_id, row["sample_id"], smell,
                        "persistence_history", "path_unreadable_at_labelled_commit", path)
        return None
    baseline_alive, baseline_holds = entity_status(baseline_source, smell, code_name)
    if not baseline_alive:
        append_failure(row["repository"], instance_id, row["sample_id"], smell,
                        "persistence_history", "entity_not_found_at_labelled_commit", code_name)
        return None

    subsequent = [(labelled_commit, label_date, path)] + forward  # revision 0 + forward walk

    def parse_git_date(s: str) -> datetime:
        # git's --date=iso-strict renders an exact-UTC offset as a trailing
        # 'Z' rather than '+00:00' on some git versions; Python 3.9's
        # datetime.fromisoformat() (this project's pinned interpreter)
        # doesn't accept 'Z' -- only fixed in 3.11. Normalize before parsing.
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    def days_since_label(d: str) -> int:
        return (parse_git_date(d) - parse_git_date(label_date)).days

    n_revisions_observed = len(subsequent) - 1  # excludes revision 0 (the label itself)
    persistence_revisions = None
    persistence_days = None
    censored = True
    death_reason = None
    death_commit, death_commit_date = None, None
    last_rev_hash, last_rev_date, last_rev_path = subsequent[-1]

    for i, (h, d, p) in enumerate(subsequent):
        if i == 0:
            continue  # baseline already verified above; count starts at revision 1
        source = read_blob(repo_dir, h, p)
        if source is None:
            death_reason = "path_unreadable_at_revision"
        else:
            alive, holds = entity_status(source, smell, code_name)
            if not alive:
                death_reason = "entity_not_found"
            elif holds is False:
                death_reason = "condition_no_longer_holds"
        if death_reason is not None:
            persistence_revisions = i
            persistence_days = days_since_label(d)
            death_commit, death_commit_date = h, d
            censored = False
            break

    if censored:
        persistence_revisions = n_revisions_observed
        persistence_days = days_since_label(last_rev_date)
        death_reason = "right_censored_at_last_observed_revision"
        death_commit, death_commit_date = last_rev_hash, last_rev_date

    # committer dates should be monotonic along commit_hash..HEAD, but a
    # rebase that preserves the original committer date, or clock skew, can
    # still (rarely) produce a negative elapsed-day count. Never silently
    # clamp or discard this -- flag it and keep the raw value so it's a
    # visible, auditable data point rather than a fabricated one.
    days_anomalous = persistence_days is not None and persistence_days < 0

    return {
        "instance_id": instance_id, "sample_id": row["sample_id"], "smell": smell,
        "severity_consensus": row["severity_consensus"], "repository": row["repository"],
        "path_at_label": path, "code_name": code_name,
        "original_commit_hash": original_commit,
        "commit_remapped": commit_remapped,
        "effective_commit_hash": labelled_commit,
        "head_sha_used": head_sha,
        "n_revisions_observed": n_revisions_observed,
        "persistence_revisions": persistence_revisions, "persistence_days": persistence_days,
        "days_anomalous_negative": days_anomalous,
        "censored": censored, "death_reason": death_reason,
        "labelled_commit_date": label_date,
        "death_commit": death_commit, "death_commit_date": death_commit_date,
        "last_observed_commit": last_rev_hash, "last_observed_commit_date": last_rev_date,
        "proxy_method": "entity_alive+size_shape_proxy_v1",
    }


def resolve_effective_commit(repo_dir: Path, commit_hash: str, a_sample_path: str, head_sha: str):
    """Once per (repo, commit_hash) -- all instances in a repo share the
    same labelled commit (confirmed: one commit_hash per repo). Returns
    (effective_commit, remapped, failure_reason_or_None)."""
    if is_ancestor(repo_dir, commit_hash, head_sha):
        return commit_hash, False, None
    remapped = find_remapped_commit(repo_dir, commit_hash, a_sample_path.lstrip("/"))
    if remapped is not None:
        # the remap match was found via `git log --all` (any branch/tag);
        # still require it to actually be an ancestor of the pinned HEAD
        # we're walking forward to, or `effective_commit..head_sha` isn't
        # a meaningful "forward" range.
        if is_ancestor(repo_dir, remapped, head_sha):
            return remapped, True, None
        return None, False, "remap_candidate_not_ancestor_of_head"
    return None, False, "commit_not_ancestor_of_head"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-list", type=Path)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    inst = pd.read_csv(INSTANCES_CSV)
    if args.repo_list:
        repos = set(l.strip() for l in args.repo_list.read_text().splitlines() if l.strip())
        inst = inst[inst["repository"].isin(repos)]
    elif not args.all:
        print("Pass --repo-list <file> or --all", file=sys.stderr)
        sys.exit(1)

    done_ids = set()
    if OUT_CSV.exists():
        done_ids = set(pd.read_csv(OUT_CSV)["instance_id"])

    write_header = not OUT_CSV.exists()
    f = open(OUT_CSV, "a", newline="")
    w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
    if write_header:
        w.writeheader()

    total = len(inst)
    n_ok, n_fail, n_skip, n_remapped = 0, 0, 0, 0
    i = 0
    t_start = time.time()
    for repository, group in inst.groupby("repository", sort=False):
        repo_dir = local_dir_for(repository)
        if not repo_dir.exists():
            for _, row in group.iterrows():
                i += 1
                if row["instance_id"] in done_ids:
                    n_skip += 1
                    continue
                append_failure(repository, row["instance_id"], row["sample_id"], row["smell"],
                                "persistence_history", "repo_not_cloned", str(repo_dir))
                n_fail += 1
            continue

        commit_hash = group.iloc[0]["commit_hash"]
        a_path = group.iloc[0]["path"]
        try:
            head_sha = resolve_head_sha(repo_dir)
            effective_commit, remapped, reason = resolve_effective_commit(repo_dir, commit_hash, a_path, head_sha)
        except Exception as e:
            effective_commit, remapped, reason = None, False, f"resolve_commit_error: {e}"

        if remapped:
            n_remapped += 1
            print(f"  [remap] {repository}: {commit_hash[:10]} -> {effective_commit[:10]} "
                  f"(matched author+date+message, verified identical file content)")

        for _, row in group.iterrows():
            i += 1
            if row["instance_id"] in done_ids:
                n_skip += 1
                continue
            if effective_commit is None:
                append_failure(repository, row["instance_id"], row["sample_id"], row["smell"],
                                "persistence_history", reason, commit_hash)
                n_fail += 1
                continue
            try:
                result = process_instance(row, repo_dir, effective_commit, remapped, head_sha)
            except Exception as e:
                append_failure(repository, row["instance_id"], row["sample_id"], row["smell"],
                                "persistence_history", "unexpected_error", e)
                result = None
            if result is None:
                n_fail += 1
            else:
                w.writerow(result)
                f.flush()
                n_ok += 1
            if i % 25 == 0 or i == total:
                elapsed = time.time() - t_start
                print(f"[{i}/{total}] ok={n_ok} fail={n_fail} skip={n_skip} remapped_repos={n_remapped} elapsed={elapsed:.1f}s")

    f.close()
    print(f"\nDone. ok={n_ok} fail={n_fail} skip={n_skip} remapped_repos={n_remapped} total={total}")


if __name__ == "__main__":
    main()
