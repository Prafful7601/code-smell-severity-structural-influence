# METHOD

Derived datasets built from `Test Smell Dataset.xlsx` (MLCQ Code Smell
Samples, Madeyski & Lewowski, EASE 2020) for two hypotheses:

- **H2**: higher-severity smells persist across more revisions before removal.
- **H3**: co-occurring Blob + Data Class instances have higher structural
  complexity than isolated instances.

Status: **FULL RUN COMPLETE** (522 repos, 9,517 instances). Final result:
**persistence.csv 8,526/9,517 (89.6%)**, **complexity.csv 8,137/9,517
(85.5%)**. Every non-produced row is accounted for in `failures.csv` with
a stage and reason (§4) -- for both tables, success count + failure count
sums to exactly 9,517.

## 0. Environment

- Python 3.9.6 (`/usr/bin/python3`), project venv at `.venv/`, pinned
  packages in `requirements.txt` (pandas 2.2.3, numpy 2.0.2, openpyxl 3.1.5).
- git 2.50.1.
- No system Java/Maven/Homebrew were available. A portable JDK (Eclipse
  Temurin 17.0.20.1+1, macOS aarch64) was downloaded into `tools/jdk-17.0.20.1+1/`
  — self-contained, no admin rights, nothing installed system-wide.
- CK (github.com/mauricioaniche/ck), pinned to release tag **ck-0.7.0**
  (commit `54c2170`), built from source with its bundled Maven wrapper
  (`./mvnw -DskipTests package`) against the portable JDK. Jar:
  `tools/ck/target/ck-0.7.0-jar-with-dependencies.jar`.
- All random sampling (repo size estimate, pilot repo selection) used a
  fixed seed (`random.seed(42)`).

## 1. Instance table (`data/instances.csv`)

One row per (sample_id, smell) — 9,517 rows, built by `scripts/01_build_instances.py`.

- Dropped 2 exact-duplicate reviewer resubmissions (same reviewer_id,
  sample_id, smell, severity, seconds apart) before aggregating.
- **Consensus severity** per instance = majority vote (mode) of severity
  across distinct reviewers. This reproduces the project's stated
  90.4/5.4/3.4/0.8% (none/minor/major/critical) distribution exactly; the
  raw row-level distribution (one row per reviewer) is materially
  different (77.7/12.1/7.7/2.5%) because reviewers disagree often.
  7,051/9,517 instances have only 1 reviewer; the rest have 2–5.
- **39 instances had a tied mode** (2 severity levels equally frequent
  across reviewers — 37 two-way ties, 2 four-way ties). Ties are broken
  toward the **higher** severity: understating severity would bias H2
  toward the null, so ties resolve upward. This is a deliberate,
  documented choice, not a default.
- `is_from_industry_relevant_project` is constant within each repository
  (confirmed: repo-level flag, not sample-level) and preserved as-is,
  including the raw `"0,5"` string value seen for some repos; not
  otherwise used by either derived table.

## 2. Persistence (`data/persistence.csv`)

### 2.1 Operationalization (agreed with the user before running)

Re-running a full smell detector at every revision was judged too
expensive across ~9,500 instances with potentially long file histories.
Instead:

- **Revision** = a commit that touches the entity's file, found via
  `git log --follow` on the labelled `path`.
- **Entity alive** = a declaration matching `code_name`'s simple name is
  still found in the file, via a lightweight regex + brace-matching parser
  (`scripts/lib/entity_finder.py`) — not a full Java parser (see §2.3 for
  why, and its limitations).
- **Smell condition holds** (a cheap size/shape proxy, not a re-run of the
  original detector):
  - *Long Method*: method LOC ≥ 100.
  - *Blob*: class LOC ≥ 200 **and** method count ≥ 20.
  - *Data Class*: ≥ 70% of methods are trivial accessors/mutators and no
    method has non-trivial control flow.
  - *Feature Envy*: ratio of external-to-internal member references in the
    method body ≥ 1 (the weakest of the four proxies — flagged explicitly).
- **Persistence (revisions)** = count of subsequent touching-revisions from
  the labelled commit (revision 0, not counted) up to and including the
  first revision where the entity is gone or its condition proxy fails.
  A death at the very next touching-commit is recorded as
  `persistence_revisions = 1`.
- **Persistence (days)** = **committer date** difference (not author date)
  between the labelled commit and the death (or last-observed) revision.
  Author date was tried first and rejected: a branch authored before, but
  merged after, another commit can carry an EARLIER author-date timestamp
  than an ancestor's, producing negative elapsed-day counts on real data
  (observed directly in the pilot: apache/calcite instance `4128743::feature
  envy` — labelled commit author date 2019-03-18, its immediate successor's
  author date 2019-03-15). Committer date tracks when a commit actually
  landed on the branch being walked and was monotonic in every case
  checked. A `days_anomalous_negative` flag is still carried on every row
  in case a rare rebase-preserved-date or clock-skew case slips through —
  never silently clamped or dropped.
- **Right-censoring**: an instance whose entity is still alive with the
  condition still holding at the last touching-revision the repo's history
  offers is right-censored (`censored=True`); its observed lifetime is
  never treated as a completed one.

### 2.2 Commit-hash remapping (discovered during the pilot)

**This is the single most consequential finding of the pilot.** Three of
the ten pilot repos (apache/poi, apache/fop, apache/ofbiz-framework) had
their labelled `commit_hash` present as a git OBJECT in a fresh clone
(fetched, `git cat-file -t` succeeds) but **unreachable from any branch or
tag** — `git merge-base --is-ancestor <commit> HEAD` fails, and a scan of
all 29 branches and 105 tags in apache/poi's clone found none containing
it. This affected **100% of instances in all three repos** (104 of the
pilot's 147 failures before the fix below).

Diagnosis, confirmed directly on two repos:

| repo | labelled commit | reachable commit with identical author, author-date, message |
|---|---|---|
| apache/poi | `351623a86924dab9c565e08e8cecfe151522c448` | `e2111d9d269adf5e0b8605b2438d06524a89cc35` |
| apache/fop | `caced35327f6b6d6eeac2a13543f3c458e902101` | `2f92dafd21360e253cdc1ca8f114b1d063c36827` |

Same author email, same to-the-second commit timestamp, same commit
message — different hash. This is consistent with a git-hosting
re-mirror/history-regeneration event on the ASF side sometime after the
2019 MLCQ data collection (**not** universal across ASF repos: 5 of the
pilot's 8 apache/* repos were unaffected — calcite, nifi-minifi,
commons-jelly, mina-sshd, tapestry-5 all had their exact original hash
reachable from HEAD). Given **48.9% of all 522 repos are apache/\*** (255
of 522) and **13.6% are eclipse\*** (71 of 522), this could plausibly
affect a meaningful share of the full corpus — the pilot's 3-in-8 rate
among apache/* repos is not something to extrapolate precisely from n=8,
but it rules out "negligible."

**Mitigation implemented** (`find_remapped_commit` in
`scripts/03_persistence.py`): when the labelled commit is unreachable,
search all reachable commits for an exact match on
(author_email, author_date, message). If **exactly one** match exists,
**and** the file content at the instance's `path` is byte-identical
between the original and candidate commit (verified via `git show`, not
assumed), treat the candidate as the effective starting commit for the
forward walk. Every remapped instance carries `commit_remapped=True`,
`original_commit_hash`, and `effective_commit_hash` in `persistence.csv`
for full auditability — nothing is silently substituted. If zero or
multiple candidates match, or content differs, the instance is logged to
`failures.csv` as `commit_not_ancestor_of_head` with no remap attempted.

**Pilot result**: all 3 affected repos (apache/poi 48 instances, apache/fop
38, apache/ofbiz-framework 18 — 104 instances, 44.4% of the pilot) were
successfully remapped and recovered. Final pilot persistence table:
**234/234 instances (100%)**, 0 entries in failures.csv for this stage.

**Full-run result**: **22 repositories** required a remap (up from the
pilot's 3), all resolved by the same verified metadata + content-match
check — no relaxation of the verification bar was needed at scale. A
further **12 repositories** had an unreachable commit with no safe remap
(no unique metadata match, or a match whose file content didn't verify),
logged as `commit_not_ancestor_of_head` (180 instances); **1 repository**
had a remap candidate found but that candidate was *itself* not an
ancestor of the pinned HEAD, so it was correctly rejected rather than
trusted (`remap_candidate_not_ancestor_of_head`, 20 instances) — see §2.3
for why that check exists.

### 2.3 A concurrency bug caught during the pilot (and its fix)

An early pilot run of `03_persistence.py` was launched concurrently with
`04_complexity.py` against the same `cache/repos/` clones, to save wall-
clock time. This corrupted the persistence results for exactly the 3
remap-affected repos: `04_complexity.py`'s `git checkout -f <commit_hash>`
moves the shared clone's `HEAD` ref to the labelled commit; if that
checkout happened before `03_persistence.py` reached that repo, the
literal ref `"HEAD"` it used as the forward-walk endpoint now pointed at
the labelled commit itself, silently truncating every instance in that
repo to `n_revisions_observed=0, censored=True` — a plausible-looking but
entirely fabricated-by-accident result (caught by noticing all 48 poi
instances showed exactly 0 revisions, which is what led to inspecting
`HEAD` and finding it detached at the labelled commit rather than at
poi's `trunk` tip). Fixed by resolving `git rev-parse HEAD` to a
concrete SHA **once** at the start of each repo's processing
(`resolve_head_sha`) and threading that pinned SHA through instead of the
mutable ref everywhere; the SHA used is recorded per row
(`head_sha_used`) for auditability. The full run must still avoid running
`03_persistence.py` and `04_complexity.py` concurrently against the same
repo before it's been pinned by the former — the fix protects a repo
already in progress, not one `04_complexity.py` touches first.

### 2.4 Known limitations of the persistence proxy

- The entity finder is regex + brace-matching, not a real Java parser.
  An earlier implementation used a return-type-prefix regex with two
  adjacent whitespace-accepting quantifiers — a textbook catastrophic-
  backtracking shape — which hung for 10+ minutes on a single instance
  whose target method was named `equals` (a name common enough to produce
  many failing regex attempts against a large file). Replaced with a
  linear scan: find bare `name(` occurrences (a single flat regex, O(n)),
  then classify each as declaration-vs-call-site using fixed-window
  string checks only (not preceded by `.`/`::`; followed shortly by `{`
  before any `;`). A real parser (javalang/JDT) was deliberately avoided
  instead of fixed differently: it requires the whole file to parse under
  a supported grammar, and this corpus's revision histories run through
  many pre-Java-8-parser-support-cutoff and post-Java-8 files (var,
  records, switch expressions, text blocks) that would simply fail to
  parse; brace-counting degrades gracefully instead.
- Overload disambiguation uses parameter **count**, not parameter types
  (MLCQ's `code_name` does carry parameter types, e.g.
  `...#lockSourceAndCopy File|File`, but matching them against arbitrary
  historical source text reliably is not cheap). A same-named,
  same-arity sibling overload can be mismatched.
- Feature Envy's condition proxy (external/internal reference ratio) is
  the weakest of the four and is a shape heuristic, not a real
  call-graph analysis.
- `git log --follow` rename-tracking across the `labelled_commit..HEAD`
  range boundary is a known git rough edge; a rename that happens exactly
  at or near that boundary can be missed.
- `--filter=blob:none` fetches blob content lazily on first request; one
  retry with a longer timeout (60s, then 180s) absorbs an occasional slow
  fetch, but a `git show` that fails twice is logged as a genuine failure,
  not retried indefinitely.
- `code_name` parsing bug (found and fixed during the pilot): the
  no-`#` branch of the parser (constructor-style / some plain-method
  entries) originally split off the method name with
  `code_name.rsplit(" ", 1)[0]` — the LAST space in the string. MLCQ's
  parameter blob can itself contain spaces (generic bounds like
  `FunctionExpression<Function2<T, Integer, Enumerable<TResult>>>` have a
  space after every comma), so on such instances the split point landed
  inside the params, not after the method name, silently producing a
  garbage method name and `entity_not_found_at_labelled_commit`. Confirmed
  on `apache/calcite`'s `QueryableDefaults.selectManyN` instance in the
  pilot. Fixed to `code_name.split(" ", 1)[0]` (the FIRST space, always
  correct since params is defined as "everything after the first space").
  The identical bug existed in `04_complexity.py`'s copy of this parser
  and was fixed there too.

## 3. Complexity (`data/complexity.csv`)

- Per repo (exactly one commit_hash per repo): `git checkout -f
  <commit_hash>` into the already-cloned working tree, then CK run once
  over the whole checkout (`use Jars=false`, no compile classpath
  supplied — CK's partial type resolution runs without it, the common way
  CK is used on arbitrary checkouts in this literature, but CBO/DIT/RFC
  for types resolvable only through external jars can be undercounted).
- **Correction to an assumption made during the pilot**: the pilot's
  METHOD.md draft claimed checkout "works on any commit object present
  locally regardless of branch reachability," reasoning that the
  §2.2 remap issue therefore couldn't affect this table. That turned out
  to be wrong at full scale. A blobless partial clone's promisor remote
  will lazily serve a *single blob* for an unreachable commit via `git
  show` (which is what `find_remapped_commit`'s content-verification
  step relies on, and how the pilot's poi/fop/ofbiz remap was verified)
  but can refuse a *full-tree checkout* of the same commit outright —
  observed directly on apache/ctakes, apache/openjpa, and
  apache/uima-ruta (`fatal: remote error: upload-pack: not our ref
  <hash>`), all with the commit object itself present locally. Fixed by
  giving `04_complexity.py` the same remap-then-retry logic as
  `03_persistence.py` (now shared in `scripts/lib/git_utils.py`): try
  the original commit, and only on checkout failure search for and
  checkout a verified remap instead. `complexity.csv` carries the same
  `original_commit_hash` / `commit_remapped` / `effective_commit_hash`
  provenance columns as `persistence.csv` for this reason. Of the 3 repos
  this affected, apache/openjpa's checkout actually succeeded on a bare
  retry of the *original* hash (no remap needed — most likely a
  transient network issue in the first run's promisor fetch, not a true
  unreachable commit); apache/ctakes and apache/uima-ruta are the same 2
  of the 12 repos from §2.2 with no safe remap available at all, and
  remain genuine attrition in both tables.
- class.csv / method.csv output cached per repo under `cache/ck/<repo>/`;
  skipped on re-run unless `--force`.
- **Join**: class smells (blob, data class) join CK's `class` (FQCN)
  column against `code_name`, cross-checked by repo-relative `file` path
  when the FQCN match is ambiguous. Function smells (feature envy, long
  method) parse `code_name` into (class FQCN, method simple name,
  parameter count) — same parsing convention as §2.1's entity finder —
  and join CK's `method` column (format `name/arity` or
  `name/arity[type1,type2,...]`) on FQCN + name, preferring an exact
  arity match.
- **Nested-class name mismatch** (found and fixed during the pilot): CK
  reports nested/inner classes using the JVM binary name separator
  (`Outer$Inner`), while MLCQ's `code_name` always uses source-level dot
  notation (`Outer.Inner`) — e.g. CK's
  `org.apache.calcite.materialize.Lattice$Measure` vs MLCQ's
  `org.apache.calcite.materialize.Lattice.Measure`. This was the majority
  cause of join misses before the fix (`class.str.replace("$", ".")`
  before comparing), applied to both the class-smell join and the class
  portion of the method-smell join.
- **Pilot result**: 230/234 instances joined (98.3%), all via exact
  `class_fqcn` or `method_fqcn+name+arity` match, no ambiguous/fallback
  joins needed.
- **Full-run result**: **8,137/9,517 instances joined (85.5%)** among
  instances whose repo cloned, checked out, and ran CK successfully —
  4,053 class-smell rows (all exact `class_fqcn`), 4,084 function-smell
  rows (all exact `method_fqcn+name+arity`); still no ambiguous/fallback
  join needed at full scale. 88 instances across 31 repos hit
  `no_join_match` (the same inherited/non-declared-method pattern seen in
  the pilot's directory-kerby example) and are logged, not imputed.
- **Pilot timings** (10 repos, CK cache cold): checkout 0.5–7.9s/repo,
  CK run 1.0–9.7s/repo; total 80.6s for all 10 repos including 3 repos
  needing a full working-tree checkout well over 100MB (poi, fop,
  ofbiz-framework). See §5 for the full-scale disk/time projection.

## 4. Attrition (`failures.csv`)

Every instance that could not get a row in either output table is logged
with a `stage`, `reason`, and `detail`. No value is ever imputed,
interpolated, or estimated — a missing cell is left NULL and the instance
is counted in `failures.csv` instead.

**Pilot result**: 4 of 468 possible (instance × table) rows failed —
persistence 0/234, complexity 4/234 — all `no_join_match` on
apache/directory-kerby, detailed in §3.

**Full-run result** (9,517 instances × 2 tables = 19,034 possible rows;
8,526 + 8,137 = 16,663 produced, **2,371 (12.5%) legitimate failures**,
exactly accounted for: success count + failure count sums to precisely
9,517 for each table, checked programmatically after the run):

| stage | reason | instances | notes |
|---|---|---:|---|
| clone | `clone_failed` | 21 repos | genuine 404s — mostly Eclipse repos that moved GitHub orgs post-2019 (`eclipse/jgit`→`eclipse-jgit/jgit` etc.), a few retired Apache Attic projects |
| persistence | `repo_not_cloned` | 750 | cascades from the 21 dead repos |
| persistence | `commit_not_ancestor_of_head` | 180 | unreachable commit, no safe remap found (§2.2) |
| persistence | `remap_candidate_not_ancestor_of_head` | 20 | remap candidate found but itself unreachable — correctly rejected, not trusted (§2.2) |
| persistence | `path_unreadable_at_labelled_commit` | 18 | path/commit combination doesn't resolve to readable content |
| persistence | `unexpected_error` | 12 | uncaught exceptions, one per distinct cause |
| persistence | `entity_not_found_at_labelled_commit` | 10 | code_name doesn't resolve in the labelled commit's own content |
| persistence | `git_log_failed` | 1 | |
| complexity | `repo_not_cloned` | 750 | same 21 dead repos |
| complexity | `ck_failed` | 486 | CK/Eclipse JDT crashes (`NullPointerException` etc.) on 8 repos — large or unusual codebases (SapMachine — an OpenJDK fork, 274 instances alone; j2objc, openj9, ceylon, error-prone-javac, reddeer, reactor-core, incubator-netbeans-html4j). A third-party tool limitation, not something this pipeline can fix. |
| complexity | `no_join_match` | 88 | across 31 repos, same inherited/non-declared-method pattern as the pilot (§3) |
| complexity | `checkout_failed` | 56 | 2 repos (apache/ctakes, apache/uima-ruta) — the same unreachable-commit issue as persistence, but breaking full-tree checkout rather than history-walking (found only at full scale; see §3's correction) |

A third repo hitting the checkout variant of this issue,
apache/openjpa (22 instances), was recovered without needing a remap at
all — its checkout succeeded on a bare retry, most likely a transient
promisor-fetch network issue in the original run rather than a true
unreachable commit.

`failures.csv` is deduplicated: clone-stage rows (repo-level, no
`instance_id`) by `repository`; persistence/complexity rows by
`(instance_id, stage)`, keeping the latest attempt, and any row whose
instance later succeeded in a retry is removed rather than left stale.
Intermediate debug runs from earlier in development (concurrency bug,
regex-hang bug, code_name parsing bug — see §2.3 and §2.4/§3) are kept
under `logs/failures_debug_run*.csv` for the record rather than deleted.

## 5. Disk and time cost (pilot-measured, then actual for the full run)

10-repo pilot projections vs. what the full 522-repo run actually used:

| stage | pilot projection | actual (522 repos) |
|---|---|---|
| blobless clone | ~7.3 GB, ~37 min | **9.4 GB**, ~9 min (git server round-trips, not CPU, dominate) |
| persistence walk | ~2.1 hours | **~1 hour** effective processing time (wall-clock across the session was longer due to interleaved unrelated work, not CPU/IO-bound) |
| complexity checkout+CK | ~55 GB, ~1.2 hours | **31 GB** (`cache/repos`) **+ 2.2 GB** (`cache/ck`) = **~33 GB**, ~99 min |
| **peak total** | ~63 GB | **~43 GB** (well under the pilot's projection — the large-repo tail was lighter than the worst-case estimate) |

Peak usage stayed well within the 273-309 GB free observed on the disk
during the run (see the session's own disk-cleanup detour, unrelated to
this pipeline). All of `cache/`, `tools/` (portable JDK + CK build), and
`.venv` are deleted after the final CSVs are produced, per the user's
request — only `data/*.csv`, `failures.csv`, `METHOD.md`, `scripts/`,
and `requirements.txt` (a few MB total) are kept.

## 6. Reproducibility

`cache/` (cloned repos + CK output), `tools/` (portable JDK + CK build),
and `.venv/` are **not** kept in this deliverable — they're multi-GB
scratch artifacts, safely reconstructible from the steps below. Only
`data/*.csv`, `failures.csv`, `METHOD.md`, `scripts/`, and
`requirements.txt` are shipped.

To reproduce from scratch:
1. `pip install -r requirements.txt` into a Python 3.9 venv.
2. Re-create `tools/` per §0 (download the pinned Temurin JDK, `git
   checkout ck-0.7.0` and build with `./mvnw -DskipTests package`).
3. `scripts/01_build_instances.py` → `data/instances.csv`
4. `scripts/02_clone_repos.py --repo-list <file>|--all` → `cache/repos/`
5. `scripts/03_persistence.py --repo-list <file>|--all` → `data/persistence.csv`
   (resumable: skips instance_ids already present)
6. Between steps 5 and 6, restore any repo whose working tree was
   checked out back to its default branch tip (`git checkout -f -B
   <default> origin/<default>`) — §2.3's concurrency bug is the reason
   this matters, and running 5 and 6 concurrently against the same
   `cache/repos/` should still be avoided even with that fix in place.
7. `scripts/04_complexity.py --repo-list <file>|--all` → `data/complexity.csv`
   (resumable: skips instance_ids already present; `--force` re-runs CK
   even if cached)

Shared logic lives in `scripts/lib/`: `entity_finder.py` (the persistence
proxy's regex/brace-matching parser, §2.4) and `git_utils.py` (the
commit-remap and blob-reading helpers used by both stages, §2.2/§3). All
stages append to the single top-level `failures.csv`.
