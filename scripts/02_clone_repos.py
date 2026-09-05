"""
02_clone_repos.py

Blobless partial clone (`--filter=blob:none`) of each repository referenced
in data/instances.csv, into cache/repos/<owner>__<repo>/. Full commit
history is needed (persistence walks history forward), so this cannot be
a shallow clone -- but blob content is fetched lazily only for objects we
actually touch later (checkouts, `git show`), which keeps it far smaller
than a full clone for large, old repos.

Resumable: skips any repo whose target dir already has a HEAD ref
(idempotent -- safe to re-run after interruption). Every attempt (success
or failure) is appended to logs/clone_log.csv so re-runs don't need to
recompute timings for repos already done, and failures are visible without
re-attempting network calls each time (use --retry-failed to force retry).

Usage:
  python scripts/02_clone_repos.py --repo-list pilot/pilot_repos.txt
  python scripts/02_clone_repos.py --all
"""
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPOS_DIR = ROOT / "cache" / "repos"
LOG_PATH = ROOT / "logs" / "clone_log.csv"
FAILURES_PATH = ROOT / "failures.csv"

LOG_FIELDS = ["repository", "local_dir", "status", "seconds", "size_bytes", "size_human", "timestamp", "error"]


def dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def local_dir_for(repo_url: str) -> Path:
    # git@github.com:owner/repo.git -> owner__repo
    owner_repo = repo_url.split(":", 1)[1].removesuffix(".git")
    safe = owner_repo.replace("/", "__")
    return REPOS_DIR / safe


def already_cloned(local_dir: Path) -> bool:
    if not (local_dir / "HEAD").exists() and not (local_dir / ".git").exists():
        return False
    # confirm it's a usable repo (has at least one ref/commit)
    r = subprocess.run(["git", "-C", str(local_dir), "rev-parse", "HEAD"],
                        capture_output=True, text=True)
    return r.returncode == 0


def append_log(row: dict, path: Path):
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def append_failure(repository: str, stage: str, reason: str, detail: str):
    fields = ["repository", "instance_id", "sample_id", "smell", "stage", "reason", "detail", "timestamp"]
    exists = FAILURES_PATH.exists()
    with open(FAILURES_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({
            "repository": repository, "instance_id": "", "sample_id": "", "smell": "",
            "stage": stage, "reason": reason, "detail": detail[:500],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })


def https_url(repo_url: str) -> str:
    # git@github.com:owner/repo.git -> https://github.com/owner/repo.git
    owner_repo = repo_url.split(":", 1)[1]
    return f"https://github.com/{owner_repo}"


def clone_one(repo_url: str, retry_failed: bool, already_logged: set) -> dict:
    local_dir = local_dir_for(repo_url)

    if already_cloned(local_dir):
        return {"repository": repo_url, "local_dir": str(local_dir), "status": "skipped_exists",
                "seconds": 0, "size_bytes": dir_size_bytes(local_dir), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "error": ""}

    if repo_url in already_logged and not retry_failed:
        return None  # previously attempted (incl. failure); don't re-hit network unless asked

    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    url = https_url(repo_url)
    t0 = time.time()
    proc = subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", url, str(local_dir)],
        capture_output=True, text=True, timeout=1800,
    )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        # clean up partial dir so a later run doesn't think it's "already cloned"
        if local_dir.exists():
            subprocess.run(["rm", "-rf", str(local_dir)])
        append_failure(repo_url, stage="clone", reason="clone_failed", detail=err)
        return {"repository": repo_url, "local_dir": str(local_dir), "status": "failed",
                "seconds": round(elapsed, 1), "size_bytes": 0, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "error": err}

    size = dir_size_bytes(local_dir)
    return {"repository": repo_url, "local_dir": str(local_dir), "status": "cloned",
            "seconds": round(elapsed, 1), "size_bytes": size, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "error": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-list", type=Path, help="file with one repo URL per line")
    ap.add_argument("--all", action="store_true", help="clone every repo in data/instances.csv")
    ap.add_argument("--retry-failed", action="store_true", help="retry repos that previously failed")
    args = ap.parse_args()

    if args.all:
        import pandas as pd
        inst = pd.read_csv(ROOT / "data" / "instances.csv")
        repos = sorted(inst["repository"].unique())
    elif args.repo_list:
        repos = [l.strip() for l in args.repo_list.read_text().splitlines() if l.strip()]
    else:
        print("Pass --repo-list <file> or --all", file=sys.stderr)
        sys.exit(1)

    already_logged = set()
    if LOG_PATH.exists():
        import pandas as pd
        prev = pd.read_csv(LOG_PATH)
        already_logged = set(prev["repository"])

    print(f"{len(repos)} repos to process.")
    for i, repo in enumerate(repos, 1):
        row = clone_one(repo, args.retry_failed, already_logged)
        if row is None:
            print(f"[{i}/{len(repos)}] {repo} -- already attempted (failed), skipping (use --retry-failed)")
            continue
        row["size_human"] = human(row["size_bytes"])
        append_log(row, LOG_PATH)
        print(f"[{i}/{len(repos)}] {repo} -- {row['status']} in {row['seconds']}s, {row['size_human']}")


if __name__ == "__main__":
    main()
