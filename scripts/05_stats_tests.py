"""
05_stats_tests.py

Real statistical tests for H1-H3, computed on this project's own full-scope
data (9,517 instances across 522 repos, all GitHub orgs -- NOT the
Apache/Eclipse-restricted N=6,907 sample used by the separate, already
-submitted "Code Smell Paper" project; the two are not interchangeable).

H1: industry-relevant projects carry a larger share of major/critical
    smells than other projects.
H2: higher-severity smells persist across more revisions before removal.
H3: co-occurring Blob + Data Class instances have higher structural
    complexity than isolated instances.

Every number here is written to outputs/stats_results.json AND printed,
so nothing gets hand-transcribed into the paper from a terminal scrollback
-- the docx edit step reads this file. No value is estimated or rounded
away silently; sample sizes (N) are reported alongside every test because
each hypothesis draws on a different-sized subset (persistence.csv is
89.6% of instances, complexity.csv is 85.5%).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SEVERITY_ORDER = ["none", "minor", "major", "critical"]
SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITY_ORDER)}


def cliffs_delta(x, y):
    """Cliff's delta: P(x>y) - P(y>x), via a O(n log n) rank-based
    computation (avoids the O(n*m) pairwise loop for large samples)."""
    x = np.sort(np.asarray(x))
    y = np.asarray(y)
    n_x, n_y = len(x), len(y)
    # for each y_j, count x_i > y_j and x_i < y_j using searchsorted on sorted x
    idx_le = np.searchsorted(x, y, side="left")   # count of x < y_j
    idx_gt = n_x - np.searchsorted(x, y, side="right")  # count of x > y_j
    greater = idx_gt.sum()  # total pairs x_i > y_j summed over j... wait see below
    less = idx_le.sum()
    # idx_gt[j] = #x > y_j; sum over j gives total (x_i, y_j) pairs with x_i > y_j
    delta = (greater - less) / (n_x * n_y)
    return float(delta)


def cohens_d(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    pooled_sd = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if pooled_sd == 0:
        return float("nan")
    return float((x.mean() - y.mean()) / pooled_sd)


def cramers_v(contingency: pd.DataFrame):
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    n = contingency.values.sum()
    r, k = contingency.shape
    v = np.sqrt((chi2 / n) / (min(r - 1, k - 1)))
    return chi2, p, dof, float(v), expected


def h1_industry_relevance(inst: pd.DataFrame, results: dict):
    print("\n" + "=" * 70)
    print("H1: industry-relevant projects carry more major/critical smells")
    print("=" * 70)

    # Binary framing per O1 ("industry-relevant projects ... and all
    # others"): raw flag "1" = relevant, "0" and "0,5" (partial) pooled as
    # "other". The 3-way breakdown is reported too, for transparency about
    # what "other" contains.
    df = inst.copy()
    df["relevance_binary"] = np.where(df["is_from_industry_relevant_project_raw"] == "1", "relevant", "other")

    ct_binary = pd.crosstab(df["severity_consensus"], df["relevance_binary"]).reindex(SEVERITY_ORDER)
    chi2, p, dof, v, expected = cramers_v(ct_binary)
    print("\nContingency table (severity x relevant/other):")
    print(ct_binary)
    print(f"\nchi2({dof}) = {chi2:.3f}, p = {p:.6g}, Cramer's V = {v:.4f}, N = {ct_binary.values.sum()}")

    min_expected = expected.min()
    print(f"Minimum expected cell count: {min_expected:.2f} "
          f"({'OK, chi-square assumptions hold' if min_expected >= 5 else 'BELOW 5 -- chi-square approximation is shaky, interpret with caution'})")

    # major+critical share, relevant vs other
    df["severe"] = df["severity_consensus"].isin(["major", "critical"])
    share = df.groupby("relevance_binary")["severe"].mean() * 100
    print("\n% major+critical by group:")
    print(share)

    ct_3way = pd.crosstab(df["severity_consensus"], df["is_from_industry_relevant_project_raw"]).reindex(SEVERITY_ORDER)
    print("\n(supplementary) 3-way raw flag breakdown:")
    print(ct_3way)

    results["H1"] = {
        "n": int(ct_binary.values.sum()),
        "contingency_binary": ct_binary.to_dict(),
        "contingency_3way_raw_flag": ct_3way.to_dict(),
        "chi2": chi2, "dof": int(dof), "p_value": p, "cramers_v": v,
        "min_expected_cell": float(min_expected),
        "pct_severe_relevant": float(share.get("relevant", float("nan"))),
        "pct_severe_other": float(share.get("other", float("nan"))),
    }


def h2_persistence_vs_severity(inst: pd.DataFrame, persist: pd.DataFrame, results: dict):
    print("\n" + "=" * 70)
    print("H2: higher-severity smells persist across more revisions")
    print("=" * 70)

    df = persist.merge(inst[["instance_id"]], on="instance_id", how="inner")
    print(f"\nN with a persistence result: {len(df)} / {len(inst)} instances "
          f"({100*len(df)/len(inst):.1f}%)")
    print(f"Right-censored: {df['censored'].sum()} ({100*df['censored'].mean():.1f}%) -- "
          f"per the pre-registered methodology, censored instances enter the comparison "
          f"at their OBSERVED persistence_revisions as a lower bound, not excluded.")

    severe = df[df["severity_consensus"].isin(["major", "critical"])]["persistence_revisions"]
    mild = df[df["severity_consensus"].isin(["minor", "none"])]["persistence_revisions"]
    print(f"\nsevere (major+critical): N={len(severe)}, median={severe.median()}, mean={severe.mean():.3f}")
    print(f"mild (minor+none): N={len(mild)}, median={mild.median()}, mean={mild.mean():.3f}")

    u_stat, p_mw = stats.mannwhitneyu(severe, mild, alternative="greater")
    delta = cliffs_delta(severe.values, mild.values)
    d = cohens_d(severe.values, mild.values)
    print(f"\nMann-Whitney U (one-sided, severe > mild): U = {u_stat:.1f}, p = {p_mw:.6g}")
    print(f"Cliff's delta = {delta:.4f}, Cohen's d (direct) = {d:.4f}")

    # per-severity-level medians/means, for the descriptive table
    by_sev = df.groupby("severity_consensus")["persistence_revisions"].agg(["count", "median", "mean", "std"])
    by_sev = by_sev.reindex(SEVERITY_ORDER)
    print("\nBy severity level:")
    print(by_sev)

    # Robustness check: log-rank test, which (unlike Mann-Whitney on raw
    # observed values) properly accounts for which values are censored
    # rather than treating a censored lower bound as if it were the true
    # value. Included because H2's data has real censoring (30%+) and a
    # published paper should not rely on a method that can understate the
    # true difference for heavily-censored groups.
    try:
        from lifelines.statistics import logrank_test
        lr = logrank_test(
            severe.values, mild.values,
            event_observed_A=(~df.loc[severe.index, "censored"]).values,
            event_observed_B=(~df.loc[mild.index, "censored"]).values,
        )
        print(f"\nRobustness check -- log-rank test (censoring-aware): "
              f"statistic = {lr.test_statistic:.3f}, p = {lr.p_value:.6g}")
        results.setdefault("H2", {})["logrank_statistic"] = float(lr.test_statistic)
        results["H2"]["logrank_p_value"] = float(lr.p_value)
    except ImportError:
        print("\n(lifelines not installed -- skipping log-rank robustness check; "
              "Mann-Whitney result above stands as the pre-registered test)")

    results.setdefault("H2", {}).update({
        "n": len(df), "n_censored": int(df["censored"].sum()),
        "pct_censored": float(100 * df["censored"].mean()),
        "n_severe": len(severe), "n_mild": len(mild),
        "median_severe": float(severe.median()), "median_mild": float(mild.median()),
        "mean_severe": float(severe.mean()), "mean_mild": float(mild.mean()),
        "mannwhitney_u": float(u_stat), "p_value": float(p_mw),
        "cliffs_delta": delta, "cohens_d": d,
        "by_severity_level": by_sev.to_dict(),
    })


def h3_cooccurrence_complexity(inst: pd.DataFrame, complexity: pd.DataFrame, results: dict):
    print("\n" + "=" * 70)
    print("H3: co-occurring Blob+Data Class have higher structural complexity")
    print("=" * 70)

    class_inst = inst[inst["type"] == "class"].copy()
    wide = class_inst.pivot(index="sample_id", columns="smell", values="severity_consensus")
    wide["blob_present"] = wide["blob"] != "none"
    wide["dc_present"] = wide["data class"] != "none"
    wide["group"] = np.select(
        [wide["blob_present"] & wide["dc_present"],
         wide["blob_present"] ^ wide["dc_present"]],
        ["co_occurring", "isolated"],
        default="neither",
    )
    print("\nSample groups (class-type samples, by presence of blob/data class as a smell):")
    print(wide["group"].value_counts())

    # join structural metrics: for co-occurring samples, average the two
    # smells' rows (same class, same commit -- CK metrics are identical
    # per class regardless of which smell instance row references it, so
    # this is a de-dup, not an average of different measurements).
    metrics = ["wmc", "cbo", "dit", "noc", "rfc", "lcom", "loc", "total_methods", "total_fields"]
    comp_class = complexity[complexity["smell"].isin(["blob", "data class"])].copy()
    per_sample = comp_class.groupby("sample_id")[metrics].mean()  # identical values per smell -> mean == the value

    merged = wide[["group"]].join(per_sample, how="inner")
    merged = merged[merged["group"].isin(["co_occurring", "isolated"])]
    print(f"\nN with complexity data joined: {len(merged)} "
          f"(co_occurring={{(merged.group=='co_occurring').sum()}}, "
          f"isolated={(merged.group == 'isolated').sum()})".replace("{(merged.group=='co_occurring').sum()}", str((merged.group == "co_occurring").sum())))

    results.setdefault("H3", {})["n_by_group"] = merged["group"].value_counts().to_dict()

    per_metric = {}
    for m in metrics:
        co = merged.loc[merged["group"] == "co_occurring", m].dropna()
        iso = merged.loc[merged["group"] == "isolated", m].dropna()
        if len(co) < 2 or len(iso) < 2:
            continue
        u, p = stats.mannwhitneyu(co, iso, alternative="greater")
        d = cohens_d(co.values, iso.values)
        per_metric[m] = {
            "n_co": len(co), "n_iso": len(iso),
            "median_co": float(co.median()), "median_iso": float(iso.median()),
            "mean_co": float(co.mean()), "mean_iso": float(iso.mean()),
            "mannwhitney_u": float(u), "p_value": float(p), "cohens_d": d,
        }
        print(f"\n{m}: co-occurring median={co.median():.2f} (n={len(co)}), "
              f"isolated median={iso.median():.2f} (n={len(iso)}), "
              f"Mann-Whitney p={p:.4g}, Cohen's d={d:.3f}")

    results["H3"]["per_metric"] = per_metric


def main():
    inst = pd.read_csv(ROOT / "data" / "instances.csv")
    persist = pd.read_csv(ROOT / "data" / "persistence.csv")
    complexity = pd.read_csv(ROOT / "data" / "complexity.csv")

    results = {}
    h1_industry_relevance(inst, results)
    h2_persistence_vs_severity(inst, persist, results)
    h3_cooccurrence_complexity(inst, complexity, results)

    with open(OUT / "stats_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nWrote {OUT / 'stats_results.json'}")


if __name__ == "__main__":
    main()
