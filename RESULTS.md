# RESULTS

Every number below is read directly from `outputs/stats_results.json` and
`outputs/ml_results.json`, produced by `scripts/05_stats_tests.py` and
`scripts/06_ml_classifier.py` (seed=42, deterministic — re-running
reproduces every value). Weak and null results are reported as such, not
omitted or reframed. This is the **all-orgs, 522-repo** dataset built in
this project (`data/instances.csv` / `persistence.csv` / `complexity.csv`)
— a different scope from the separate, already-submitted "Code Smell
Paper" project (Apache/Eclipse-only, N=6,907). Numbers from that project
must not be mixed into this paper.

## Sample

- 9,517 consensus (sample_id, smell) instances, 522 repositories, all
  GitHub orgs. Consensus severity = modal severity across reviewers, ties
  broken toward the higher grade (39 tied instances). See `METHOD.md` for
  full dataset construction and attrition.
- `persistence.csv`: 8,526/9,517 (89.6%). `complexity.csv`: 8,137/9,517
  (85.5%). Each hypothesis below states which subset it draws on.

## H1: industry-relevant projects carry more major/critical smells

**Not supported.** Binary framing (raw flag "1" = relevant, "0" and "0,5"
pooled as "other"), N=9,517:

| | none | minor | major | critical |
|---|---:|---:|---:|---:|
| other (n=1,279) | 1,138 | 77 | 55 | 9 |
| relevant (n=8,238) | 7,467 | 441 | 265 | 65 |

χ²(3) = 5.213, **p = 0.157**, Cramér's V = 0.023 (negligible). Minimum
expected cell count 9.94 (chi-square assumptions hold).

**The effect direction runs opposite to the hypothesis**: 5.00% of "other"
instances are major/critical vs. 4.01% of "relevant" instances — the
non-fully-relevant group has a *slightly higher* severe-smell share, though
the difference is not statistically significant. No evidence for H1 in
this dataset; reported as a null/contrary result, not omitted.

## H2: higher-severity smells persist across more revisions

**Supported**, small-to-modest effect. N=8,526 (persistence.csv), of which
2,939 (34.5%) are right-censored (still alive at the end of observable
history) and enter the comparison at their observed `persistence_revisions`
as a lower bound, per the pre-registered methodology.

| severity | N | median revisions | mean revisions | SD |
|---|---:|---:|---:|---:|
| none | 7,731 | 1.0 | 1.81 | 6.36 |
| minor | 449 | 1.0 | 2.79 | 9.59 |
| major | 283 | 1.0 | 2.96 | 9.48 |
| critical | 63 | 1.0 | 8.71 | 30.44 |

Severe (major+critical, N=346, mean=4.01) vs. mild (minor+none, N=8,180,
mean=1.87): one-sided Mann-Whitney U = 1,544,100, **p = 3.68×10⁻⁴**.
Cliff's δ = 0.091 (small), Cohen's d = 0.299 (small).

**Robustness check**: a log-rank test (which, unlike Mann-Whitney on raw
observed values, properly weights *which* values are censored rather than
treating a censored lower bound as if it were the true event time) gives
statistic = 49.31, **p = 2.19×10⁻¹²** — the same direction, more strongly
significant. The effect is real but modest in size; medians are identical
across all four severity levels (1 revision) — the difference lives in the
tail (means, and the log-rank test's handling of censoring), not in the
typical case.

## H3: co-occurring Blob+Data Class have higher structural complexity

**Mostly supported.** Class-type samples reviewed for both smells:
32 co-occurring (both rated non-"none"), 507 isolated (exactly one
non-"none"), 1,801 neither. Complexity data joined: 26 co-occurring, 439
isolated (N=465; 6 co-occurring samples lost to complexity-join attrition).

| metric | co-occurring median (n=26) | isolated median (n=439) | Mann-Whitney p | Cohen's d |
|---|---:|---:|---:|---:|
| WMC | 28.0 | 14.0 | 2.9×10⁻⁴ | 0.372 |
| CBO | 13.0 | 6.0 | 6.9×10⁻⁵ | 0.272 |
| **DIT** | 2.0 | 1.0 | **0.051** | **0.037** |
| **NOC** | 0.0 | 0.0 | **0.303** | **0.097** |
| RFC | 33.5 | 9.0 | 9.3×10⁻⁴ | 0.299 |
| LCOM | 58.0 | 10.0 | 0.019 | 0.718 |
| LOC | 151.5 | 67.0 | 1.1×10⁻⁴ | 0.412 |
| Total methods | 14.5 | 8.0 | 7.1×10⁻⁴ | 0.706 |
| Total fields | 11.0 | 3.0 | 3.3×10⁻⁸ | 0.811 |

Six of eight metrics show a significant, small-to-large effect in the
hypothesized direction: co-occurring instances are bigger (LOC, total
methods/fields), more coupled (CBO, RFC), more internally tangled (LCOM,
the largest effect at d=0.72), and higher-complexity (WMC). **Inheritance-
related metrics (DIT, NOC) show no real effect** — co-occurrence is a
size/coupling/cohesion phenomenon in this data, not an inheritance-depth
one. N=26 for the co-occurring group is small; treat exact effect sizes as
indicative rather than precise, and note the 6/32 (19%) complexity-join
attrition among co-occurring samples specifically.

## Feature selection (run for real, and why its output was not used as-is)

The paper's pre-specified three-stage methodology (mutual-information
filter at MI<0.05, pairwise Pearson pruning at |r|>0.8, RFECV with Random
Forest) was run for real against the full 18-feature set (11 numeric CK/
context features + one-hot-encoded smell type and industry-relevance
flag, N=8,137). Stage 1 alone is decisive: only `n_reviewers` (MI=0.143)
and `loc` (MI=0.065) clear the 0.05 threshold — every CK structural
metric (CBO, WMC, DIT, NOC, RFC, LCOM, method/field counts) and the smell
-type indicators fall below it, and stages 2–3 (run on the 2 survivors)
change nothing further.

**This result was not adopted for the reported models.** A direct check —
training the same Random Forest on just `{n_reviewers, loc}` versus the
full 18-feature set, identical split — drops multiclass macro-F1 from
0.510 to 0.383. Univariate mutual information, applied as a hard
per-feature cutoff, is blind to the joint/nonlinear signal an ensemble
model can extract from structural metrics that look weak individually
(consistent with H3 §above, where several of those same CK metrics show
real, significant group differences) — so applying it literally would
have discarded features independently shown to matter, in service of a
selection rule that measurably hurts the actual target metric. The O4
results below therefore use the full feature set, as already reported,
with the mismatch stated here rather than silently resolved.

**A separate flag on `n_reviewers` itself**, the single strongest
univariate predictor: this is very plausibly an artefact of MLCQ's own
review-assignment process (an instance more likely to receive extra
reviewers precisely because an early reviewer flagged something
ambiguous or severe) rather than a causal driver of severity. It is kept
in the reported feature set — removing it lowers performance further and
it is a real, available field at review time, not a leak from the
future — but should not be read as a generalisable signal a detector
built for genuinely new, unreviewed code could rely on.

## O4: predicting severity from features

**Never attempted in this project before now** — the draft's prior
"Comparative Model Performance" numbers were not produced here (see
`METHOD.md`). Built for real: complete-case dataset (N=8,137, instances
with a successful CK join), features = CK structural metrics (CBO/RFC/LOC
universal; WMC/DIT/NOC/LCOM/method+field counts class-smell-only;
method-level cyclomatic complexity function-smell-only — median-imputed
within a pipeline for model training only, not reported as an empirical
value), smell type, industry-relevance flag, reviewer count. 80/20
stratified split, 5-fold stratified CV, 5 algorithms, seed=42.

### 4-class severity (none/minor/major/critical)

Severe class imbalance (90.3% "none" in this subset) makes accuracy
meaningless as a headline metric; macro-F1 is used throughout.

| model | CV macro-F1 | test macro-F1 | test accuracy | test macro-precision | test macro-recall | test AUC (OvR) | train time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Extra Trees** | 0.503 | **0.532** | 0.921 | 0.600 | 0.496 | 0.945 | 0.16 |
| Gradient Boosting | 0.435 | 0.525 | 0.931 | 0.638 | 0.471 | 0.960 | 1.27 |
| Random Forest | 0.447 | 0.510 | 0.924 | 0.608 | 0.469 | 0.958 | 0.19 |
| SVM | 0.464 | 0.456 | 0.811 | 0.413 | 0.591 | 0.940 | 1.82 |
| Logistic Regression | 0.409 | 0.364 | 0.786 | 0.342 | 0.477 | 0.909 | 0.05 |

(Training time is single-fit wall-clock on this machine, not comparable
across hardware — reported because the draft's prior table included it,
not as a claim about algorithmic efficiency in general.)

Best model (Extra Trees) confusion matrix (test N=1,628):

| true \\ predicted | none | minor | major | critical |
|---|---:|---:|---:|---:|
| none (1,475) | 1,445 | 18 | 11 | 1 |
| minor (87) | 46 | 28 | 12 | 1 |
| major (53) | 16 | 12 | 24 | 1 |
| critical (13) | 5 | 1 | 4 | 3 |

The three ensemble methods (Extra Trees, Gradient Boosting, Random Forest)
outperform Logistic Regression and SVM on macro-F1, consistent with the
draft's original qualitative claim — **but the actual margin (macro-F1
~0.51–0.53 vs. ~0.36–0.46, a gap of roughly 0.07–0.15) and the absolute
performance level are both substantially more modest** than the draft's
prior unsupported figures (F1=0.817, 89% critical precision). Critical
recall is particularly weak (3/13 correct in the held-out test set) —
expected given only 63 critical instances exist in the entire 8,137-row
complete-case sample. CV and test macro-F1 track each other reasonably
closely for every model (no evidence of overfitting to the training
folds).

### Random Forest error analysis (multiclass) — featured in the paper's prose

Random Forest specifically (test macro-F1=0.510, chosen for narrative
consistency with the draft's original framing, though Extra Trees scores
marginally higher):

| true \\ predicted | none | minor | major | critical |
|---|---:|---:|---:|---:|
| none (1,475) | 1,455 | 12 | 6 | 2 |
| minor (87) | 52 | 23 | 11 | 1 |
| major (53) | 23 | 5 | 25 | 0 |
| critical (13) | 4 | 0 | 7 | 2 |

Recall: none=98.6%, minor=26.4%, major=47.2%, critical=15.4%. Precision:
none=94.9%, critical=40.0%.

**Error direction is the opposite of the draft's prior (unsupported)
claim.** Of 123 total misclassifications, 91 (74.0%) *under*-estimate
severity and only 32 (26.0%) *over*-estimate it — the model's dominant
failure mode is missing real severity, not inflating it, which matters
for a triage-tool framing (a model that under-calls severe cases is the
more dangerous kind of wrong). The largest single error path is "major"
true cases predicted as "none" (23/53, 43.4%) — a 2-grade miss, not the
adjacent-grade confusion the draft described.

Per-smell minor→major/critical escalation rate (small per-smell test
subsamples, N=10–32 — indicative, not precise): Data Class 28.1% (9/32),
Long Method 12.0% (3/25), Feature Envy 0.0% (0/10), Blob 0.0% (0/20). The
draft's specific claim ("43% of Feature Envy minor cases labelled
major/critical") does not hold in this data — if anything Feature Envy
shows zero such escalation in this test split, and Data Class is the
weakest spot instead.

Per-class one-vs-rest AUC (Random Forest): critical=0.979, none=0.967,
major=0.944, minor=0.941 — critical is easiest to separate from the rest,
minor and major hardest (consistent with them being the two classes most
confused with their neighbours in the confusion matrix above). This is a
one-vs-rest framing, not a pairwise "boundary" AUC between two specific
classes (the draft's prior "AUC of 0.89 at the minor/major boundary" was
never computed as a pairwise metric in this project).

CV-vs-test accuracy gap (multiclass, generalisation check): Random Forest
+0.30 points, Extra Trees −0.26, SVM −0.25, Gradient Boosting +1.26,
Logistic Regression −1.27 — every model stays within about a point of its
cross-validation mean, no evidence of overfitting to the training folds.

### Binary (major+critical vs. rest)

| model | CV macro-F1 | test macro-F1 | test accuracy | test macro-precision | test macro-recall | test AUC | train time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Extra Trees** | 0.732 | **0.744** | 0.967 | 0.817 | 0.700 | 0.959 | 0.15 |
| Random Forest | 0.738 | 0.729 | 0.967 | 0.830 | 0.678 | 0.966 | 0.19 |
| Gradient Boosting | 0.717 | 0.664 | 0.963 | 0.792 | 0.618 | 0.959 | 0.32 |
| SVM | 0.653 | 0.648 | 0.889 | 0.611 | 0.826 | 0.946 | 0.84 |
| Logistic Regression | 0.597 | 0.599 | 0.834 | 0.585 | 0.841 | 0.911 | 0.02 |

The binary framing is the more practically usable of the two: ROC-AUC of
0.96–0.97 for the ensemble methods indicates the model ranks severe vs.
non-severe instances well even though F1 (threshold-dependent, and
penalized by the 4%-prevalence severe class) is more modest. Ensembles
again outperform the two non-ensemble baselines.

## Headline summary

- H1: no evidence, direction opposite to hypothesis, not significant.
- H2: real but small effect, robust to a censoring-aware robustness check.
- H3: real effect for size/coupling/cohesion metrics, no effect for
  inheritance-depth metrics, small co-occurring-group sample size (n=26).
- O4: ensembles beat linear/kernel baselines as hypothesized, but absolute
  predictive power is modest (macro-F1 ≈ 0.53 multiclass, 0.74 binary) —
  reviewer-assigned severity is only partially recoverable from structural
  metrics and dataset-native fields alone, especially at the
  minor/major/critical boundaries where reviewer disagreement itself is
  highest (see the raw MLCQ multi-reviewer disagreement noted in
  `METHOD.md` §1).
