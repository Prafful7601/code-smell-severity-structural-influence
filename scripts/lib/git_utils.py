"""
git_utils.py

Shared git helpers used by both 03_persistence.py and 04_complexity.py.
Extracted after the same commit-hash-remapping issue was found breaking
both scripts in different ways: 03_persistence.py's forward-history walk
needs an ancestor-of-HEAD commit range, and 04_complexity.py's `git
checkout -f <commit_hash>` fails outright ("not our ref") on a commit
that's present as an object locally but unreachable from any ref -- the
partial-clone promisor remote will lazily serve a single blob via `git
show` for such a commit (used by the remap verification below) but
refuses to serve a full-tree checkout of it. See METHOD.md.
"""
import subprocess
from pathlib import Path


def run_git(repo_dir: Path, args: list, timeout=60) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo_dir)] + args, capture_output=True, text=True, timeout=timeout)


def resolve_head_sha(repo_dir: Path) -> str:
    """Pin the default branch tip to a concrete SHA once, rather than using
    the literal ref "HEAD" throughout a repo's processing. This clone's
    working tree is shared between scripts -- if one does a `git checkout`
    while another is mid-walk using the literal "HEAD" ref (as happened in
    an earlier pilot run: see METHOD.md), the walk's endpoint silently
    moves underneath it. Pinning removes the shared-mutable-ref hazard and
    makes each run's boundary an auditable fixed commit."""
    proc = run_git(repo_dir, ["rev-parse", "HEAD"])
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return proc.stdout.strip()


def is_ancestor(repo_dir: Path, commit: str, of: str) -> bool:
    proc = run_git(repo_dir, ["merge-base", "--is-ancestor", commit, of])
    return proc.returncode == 0


def read_blob(repo_dir: Path, commit_hash: str, path: str):
    # blobless partial clone fetches this blob lazily over the network on
    # first request; one retry with a longer timeout absorbs an occasional
    # slow/flaky fetch without masking a genuinely missing/unreachable blob.
    for timeout in (60, 180):
        try:
            proc = run_git(repo_dir, ["show", f"{commit_hash}:{path}"], timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
        if proc.returncode != 0:
            return None
        return proc.stdout
    raise RuntimeError(f"git show timed out twice (60s, 180s) for {commit_hash}:{path}")


def find_remapped_commit(repo_dir: Path, commit_hash: str, sample_path: str):
    """When commit_hash exists as an object in the local repo (git fetched
    it) but is unreachable from any branch/tag, some repos in this corpus
    have undergone a git-hosting re-mirror that re-wrote commit hashes
    while preserving author identity, author date, and message -- verified
    empirically for apache/poi and apache/fop in this project's pilot
    (identical author/date/message, different hash; see METHOD.md). This
    searches all reachable commits for that exact (author_email,
    author_date, message) triple, then requires the file content at
    `sample_path` to be byte-identical between the original and candidate
    commit before trusting the match -- never remaps on metadata alone.
    Returns the candidate hash, or None if no safe unique remap exists."""
    meta = run_git(repo_dir, ["log", "-1", "--format=%ae|%ad|%s", "--date=iso-strict", commit_hash])
    if meta.returncode != 0 or "|" not in meta.stdout:
        return None
    try:
        author_email, author_date, message = meta.stdout.rstrip("\n").split("|", 2)
    except ValueError:
        return None

    search = run_git(repo_dir, ["log", "--all", "--format=%H|%ae|%ad|%s", "--date=iso-strict"], timeout=300)
    if search.returncode != 0:
        return None

    candidates = []
    for line in search.stdout.splitlines():
        parts = line.split("|", 3)
        if len(parts) != 4:
            continue
        h, ae, ad, msg = parts
        if h != commit_hash and ae == author_email and ad == author_date and msg == message:
            candidates.append(h)

    if len(candidates) != 1:
        return None

    candidate = candidates[0]
    original_blob = read_blob(repo_dir, commit_hash, sample_path)
    candidate_blob = read_blob(repo_dir, candidate, sample_path)
    if original_blob is None or candidate_blob is None or original_blob != candidate_blob:
        return None
    return candidate


