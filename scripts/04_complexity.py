"""
04_complexity.py

For each (sample_id, smell) instance, structural metrics AT THE LABELLED
COMMIT: WMC, CBO, DIT, NOC, RFC, LCOM, cyclomatic complexity, LOC, number
of methods, number of fields -- via the CK tool (mauricioaniche/ck,
pinned to release tag ck-0.7.0, built from source with its Maven wrapper
against a locally-installed portable JDK 17; see METHOD.md).

For each repository (recall: exactly one commit_hash per repo):
  1. Force-checkout that commit into the already-cloned working tree
     (cache/repos/<repo>/). This materializes the full tree on disk --
     the one place in this pipeline where disk cost is NOT bounded by
     "only the files we touch" (CK needs the whole project on disk to
     resolve types across files).
  2. Run CK once over the whole checked-out tree -> class.csv + method.csv
     (cached under cache/ck/<repo>/; skipped on re-run unless --force).
  3. Join back to instances on (path, code_name):
       - class smells (blob, data class): join on CK's `class` (FQCN)
         column, cross-checked against `file` (repo-relative path).
       - function smells (feature envy, long method): parse code_name into
         (class FQCN, method simple name, param count) exactly as
         03_persistence.py does, and join on CK's method.csv
         `class`==FQCN and `method` matching `name/paramCount[...]`.
  No compile classpath / dependency jars are supplied to CK ("use
  Jars=false") -- CK's partial type resolution runs without them, which is
  the common way CK is used on arbitrary checkouts in this literature, but
  it means CBO/DIT/RFC for types resolved only through external jars can
  be undercounted. Documented in METHOD.md.

Resumable: skips instance_ids already in data/complexity.csv. Every
instance CK could not compute a row for (repo not cloned, checkout
failed, CK crashed, no join match) is logged to failures.csv with a
reason -- no value is ever imputed.

Usage:
  python scripts/04_complexity.py --repo-list pilot/pilot_repos.txt
  python scripts/04_complexity.py --all
"""
import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.git_utils import find_remapped_commit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPOS_DIR = ROOT / "cache" / "repos"
CK_CACHE_DIR = ROOT / "cache" / "ck"
INSTANCES_CSV = ROOT / "data" / "instances.csv"
OUT_CSV = ROOT / "data" / "complexity.csv"
FAILURES_CSV = ROOT / "failures.csv"

JDK_JAVA = ROOT / "tools" / "jdk-17.0.20.1+1" / "Contents" / "Home" / "bin" / "java"
CK_JAR = ROOT / "tools" / "ck" / "target" / "ck-0.7.0-jar-with-dependencies.jar"

FAILURE_FIELDS = ["repository", "instance_id", "sample_id", "smell", "stage", "reason", "detail", "timestamp"]

OUT_FIELDS = [
    "instance_id", "sample_id", "smell", "severity_consensus", "repository",
    "original_commit_hash", "commit_remapped", "effective_commit_hash",
    "path_at_label", "code_name", "join_level",
    "wmc", "cbo", "dit", "noc", "rfc", "lcom", "loc", "total_methods", "total_fields",
    "method_cyclomatic_complexity",  # method-level `wmc` from CK == McCabe CC of that one method; NULL for class smells
    "ck_version",
]


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


def ck_out_dir_for(repo_url: str) -> Path:
    owner_repo = repo_url.split(":", 1)[1].removesuffix(".git")
    return CK_CACHE_DIR / owner_repo.replace("/", "__")


def run_git(repo_dir: Path, args: list, timeout=300):
    return subprocess.run(["git", "-C", str(repo_dir)] + args, capture_output=True, text=True, timeout=timeout)


def checkout_commit(repo_dir: Path, commit_hash: str) -> tuple:
    t0 = time.time()
    proc = run_git(repo_dir, ["checkout", "-f", commit_hash], timeout=600)
    return proc.returncode == 0, proc.stderr.strip(), time.time() - t0


def checkout_with_remap_fallback(repo_dir: Path, commit_hash: str, a_sample_path: str) -> tuple:
    """Try checking out commit_hash directly first. A blobless-partial-clone
    checkout can fail with "not our ref" even when the commit OBJECT exists
    locally (confirmed on apache/ctakes, apache/openjpa, apache/uima-ruta in
    the full run) -- unlike a single-file `git show`, which the promisor
    remote will lazily serve for such a commit (that asymmetry is exactly
    what makes find_remapped_commit's content-verification step work at
    all). On failure, attempt the same verified remap used by
    03_persistence.py and retry the checkout with the candidate. Returns
    (ok, effective_commit, remapped, error, seconds)."""
    t0 = time.time()
    ok, err, _ = checkout_commit(repo_dir, commit_hash)
    if ok:
        return True, commit_hash, False, None, time.time() - t0

    remapped_commit = find_remapped_commit(repo_dir, commit_hash, a_sample_path.lstrip("/"))
    if remapped_commit is None:
        return False, commit_hash, False, err, time.time() - t0

    ok2, err2, _ = checkout_commit(repo_dir, remapped_commit)
    if ok2:
        return True, remapped_commit, True, None, time.time() - t0
    return False, commit_hash, False, f"original: {err} | after remap to {remapped_commit}: {err2}", time.time() - t0


def run_ck(repo_dir: Path, out_dir: Path) -> tuple:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc = subprocess.run(
        [str(JDK_JAVA), "-jar", str(CK_JAR), str(repo_dir) + "/", "false", "0", "false", str(out_dir) + "/"],
        capture_output=True, text=True, timeout=1800,
    )
    ok = proc.returncode == 0 and (out_dir / "class.csv").exists()
    return ok, (proc.stderr or "").strip()[-2000:], time.time() - t0


def parse_function_code_name(code_name: str):
    """Mirrors 03_persistence.py's entity_status parsing. Returns
    (class_fqcn, method_simple_name, n_params)."""
    if "#" in code_name:
        head, rest = code_name.split("#", 1)
        parts = rest.split(" ", 1)
        method_name = parts[0]
        params = parts[1] if len(parts) > 1 else ""
        class_fqcn = head
    else:
        # split on the FIRST space -- see the matching comment in
        # 03_persistence.py's entity_status for why rsplit is wrong here
        # (a generic-bounded params blob can itself contain spaces).
        head_dotted = code_name.split(" ", 1)[0]
        class_fqcn = ".".join(head_dotted.split(".")[:-1])
        method_name = head_dotted.split(".")[-1]
        params = code_name.split(" ", 1)[1] if " " in code_name else ""
    n_params = len([p for p in params.split("|") if p]) if params else 0
    return class_fqcn, method_name, n_params


_CK_METHOD_SIG_RE = re.compile(r"^([^/]+)/(\d+)")


def load_ck_tables(out_dir: Path):
    class_df = pd.read_csv(out_dir / "class.csv")
    # CK reports nested/inner classes with the JVM binary name separator
    # ('Outer$Inner'), MLCQ's code_name uses plain source-level dot
    # notation ('Outer.Inner') throughout. '$' -> '.' makes the two
    # directly comparable; this was the majority cause of join misses in
    # the pilot (confirmed: e.g. CK's
    # "org.apache.calcite.materialize.Lattice$Measure" vs MLCQ's
    # "org.apache.calcite.materialize.Lattice.Measure").
    class_df["_class_dotted"] = class_df["class"].str.replace("$", ".", regex=False)
    method_path = out_dir / "method.csv"
    method_df = pd.read_csv(method_path) if method_path.exists() else pd.DataFrame()
    if not method_df.empty:
        method_df["_class_dotted"] = method_df["class"].str.replace("$", ".", regex=False)
        parsed = method_df["method"].str.extract(_CK_METHOD_SIG_RE)
        method_df["_method_name"] = parsed[0]
        method_df["_method_arity"] = pd.to_numeric(parsed[1], errors="coerce")
    return class_df, method_df


def repo_relative(ck_file: str, repo_dir: Path) -> str:
    try:
        return str(Path(ck_file).resolve().relative_to(repo_dir.resolve()))
    except ValueError:
        return ck_file


def join_class_instance(row, class_df: pd.DataFrame, repo_dir: Path):
    code_name = row["code_name"]
    matches = class_df[class_df["_class_dotted"] == code_name]
    join_level = "class_fqcn"
    if matches.empty:
        return None, None
    if len(matches) > 1:
        # disambiguate by path if more than one CK entry claims this FQCN
        # (can happen with duplicate/ambiguous package-private classes)
        target_path = row["path"].lstrip("/")
        by_path = matches[matches["file"].apply(lambda f: repo_relative(f, repo_dir) == target_path)]
        if len(by_path) == 1:
            matches = by_path
            join_level = "class_fqcn+path"
        else:
            matches = matches.iloc[[0]]
            join_level = "class_fqcn_ambiguous_first_match"
    return matches.iloc[0], join_level


def join_method_instance(row, method_df: pd.DataFrame):
    if method_df.empty:
        return None, None
    class_fqcn, method_name, n_params = parse_function_code_name(row["code_name"])
    candidates = method_df[(method_df["_class_dotted"] == class_fqcn) & (method_df["_method_name"] == method_name)]
    if candidates.empty:
        return None, None
    exact = candidates[candidates["_method_arity"] == n_params]
    if len(exact) >= 1:
        return exact.iloc[0], "method_fqcn+name+arity"
    return candidates.iloc[0], "method_fqcn+name_only"


def process_repo(repo_url: str, group: pd.DataFrame, done_ids: set, force: bool):
    repo_dir = local_dir_for(repo_url)
    commit_hash = group.iloc[0]["commit_hash"]
    results = []

    if not repo_dir.exists():
        for _, row in group.iterrows():
            if row["instance_id"] in done_ids:
                continue
            append_failure(repo_url, row["instance_id"], row["sample_id"], row["smell"],
                            "complexity", "repo_not_cloned", str(repo_dir))
        return results, {"checkout_s": 0, "ck_s": 0}

    ck_out = ck_out_dir_for(repo_url)
    have_cache = (ck_out / "class.csv").exists() and not force
    timings = {"checkout_s": 0, "ck_s": 0}
    # cached CK output can only exist from a prior run that reached a
    # successful checkout -- which our checkout-then-remap-fallback always
    # attempts with the ORIGINAL commit first, so a cache hit implies the
    # original hash worked (remapped=False). A fresh run may set these to
    # a remapped commit instead; see checkout_with_remap_fallback.
    effective_commit, remapped = commit_hash, False

    if not have_cache:
        a_path = group.iloc[0]["path"]
        ok, effective_commit, remapped, err, checkout_s = checkout_with_remap_fallback(repo_dir, commit_hash, a_path)
        timings["checkout_s"] = round(checkout_s, 1)
        if not ok:
            for _, row in group.iterrows():
                if row["instance_id"] in done_ids:
                    continue
                append_failure(repo_url, row["instance_id"], row["sample_id"], row["smell"],
                                "complexity", "checkout_failed", err)
            return results, timings
        if remapped:
            print(f"  [remap] {repo_url}: {commit_hash[:10]} -> {effective_commit[:10]} "
                  f"(checkout only succeeded after remap; matched author+date+message, verified identical content)")

        ok, err, ck_s = run_ck(repo_dir, ck_out)
        timings["ck_s"] = round(ck_s, 1)
        if not ok:
            for _, row in group.iterrows():
                if row["instance_id"] in done_ids:
                    continue
                append_failure(repo_url, row["instance_id"], row["sample_id"], row["smell"],
                                "complexity", "ck_failed", err)
            return results, timings

    try:
        class_df, method_df = load_ck_tables(ck_out)
    except Exception as e:
        for _, row in group.iterrows():
            if row["instance_id"] in done_ids:
                continue
            append_failure(repo_url, row["instance_id"], row["sample_id"], row["smell"],
                            "complexity", "ck_output_unreadable", e)
        return results, timings

    for _, row in group.iterrows():
        if row["instance_id"] in done_ids:
            continue
        is_class_smell = row["smell"] in ("blob", "data class")
        if is_class_smell:
            match, join_level = join_class_instance(row, class_df, repo_dir)
        else:
            match, join_level = join_method_instance(row, method_df)

        if match is None:
            append_failure(repo_url, row["instance_id"], row["sample_id"], row["smell"],
                            "complexity", "no_join_match", f"code_name={row['code_name']!r} path={row['path']!r}")
            continue

        if is_class_smell:
            results.append({
                "instance_id": row["instance_id"], "sample_id": row["sample_id"], "smell": row["smell"],
                "severity_consensus": row["severity_consensus"], "repository": repo_url,
                "original_commit_hash": commit_hash, "commit_remapped": remapped,
                "effective_commit_hash": effective_commit,
                "path_at_label": row["path"], "code_name": row["code_name"],
                "join_level": join_level,
                "wmc": match["wmc"], "cbo": match["cbo"], "dit": match["dit"], "noc": match["noc"],
                "rfc": match["rfc"], "lcom": match["lcom"], "loc": match["loc"],
                "total_methods": match["totalMethodsQty"], "total_fields": match["totalFieldsQty"],
                "method_cyclomatic_complexity": None,
                "ck_version": "0.7.0",
            })
        else:
            results.append({
                "instance_id": row["instance_id"], "sample_id": row["sample_id"], "smell": row["smell"],
                "severity_consensus": row["severity_consensus"], "repository": repo_url,
                "original_commit_hash": commit_hash, "commit_remapped": remapped,
                "effective_commit_hash": effective_commit,
                "path_at_label": row["path"], "code_name": row["code_name"],
                "join_level": join_level,
                "wmc": None, "cbo": match["cbo"], "dit": None, "noc": None,
                "rfc": match["rfc"], "lcom": None, "loc": match["loc"],
                "total_methods": None, "total_fields": None,
                "method_cyclomatic_complexity": match["wmc"],  # CK's method-level `wmc` IS McCabe CC
                "ck_version": "0.7.0",
            })
    return results, timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-list", type=Path)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run CK even if cached output exists")
    args = ap.parse_args()

    if not JDK_JAVA.exists() or not CK_JAR.exists():
        print(f"Missing JDK ({JDK_JAVA}) or CK jar ({CK_JAR}) -- see tools/ setup steps.", file=sys.stderr)
        sys.exit(1)

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

    repos = sorted(inst["repository"].unique())
    n_ok_total, n_repo_total = 0, len(repos)
    t_start = time.time()
    for i, repo_url in enumerate(repos, 1):
        group = inst[inst["repository"] == repo_url]
        results, timings = process_repo(repo_url, group, done_ids, args.force)
        for r in results:
            w.writerow(r)
        f.flush()
        n_ok_total += len(results)
        elapsed = time.time() - t_start
        print(f"[{i}/{n_repo_total}] {repo_url} -- {len(results)}/{len(group)} joined, "
              f"checkout={timings['checkout_s']}s ck={timings['ck_s']}s, total_elapsed={elapsed:.1f}s")

    f.close()
    n_fail = len(inst) - len(done_ids) - n_ok_total
    print(f"\nDone. {n_ok_total} instances joined this run. See failures.csv for misses.")


if __name__ == "__main__":
    main()
