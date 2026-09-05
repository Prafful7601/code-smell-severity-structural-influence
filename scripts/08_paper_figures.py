"""
08_paper_figures.py

Regenerates Fig. 5 (Dataset Characteristics) and Fig. 7 (ML Algorithm
Comparison) from real data -- both existing images in the draft were
fabricated (Fig. 5's severity distribution has MORE minor+major+critical
than "none", the opposite of every real accounting of this dataset;
Fig. 7's numbers are identical to the fabricated Table I). Figs. 8, 9, 10
reuse images already produced by 06_ml_classifier.py. Figs. 1-4
(literature-review figures, not this project's data) and Fig. 6 (a
generic three-stage-selection flowchart, not data-specific) are
unchanged.
"""
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEVERITY_ORDER = ["none", "minor", "major", "critical"]


def fig5_dataset_characteristics():
    inst = pd.read_csv(ROOT / "data" / "instances.csv")

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    # (a) instances per smell type
    ax = axes[0, 0]
    counts = inst["smell"].value_counts().reindex(["feature envy", "long method", "blob", "data class"])
    bars = ax.bar([s.title() for s in counts.index], counts.values, color="#4C72B0")
    ax.set_title("(a) Instances per smell type")
    ax.bar_label(bars)
    ax.set_ylim(0, max(counts.values) * 1.15)

    # (b) severity distribution
    ax = axes[0, 1]
    sev_counts = inst["severity_consensus"].value_counts().reindex(SEVERITY_ORDER)
    colors = ["#55A868", "#DBB856", "#DD8452", "#C44E52"]
    bars = ax.bar([s.title() for s in SEVERITY_ORDER], sev_counts.values, color=colors)
    ax.set_title("(b) Severity distribution")
    ax.set_yscale("log")
    ax.bar_label(bars)

    # (c) industry relevance
    ax = axes[1, 0]
    relevant = (inst["is_from_industry_relevant_project_raw"] == "1").sum()
    other = len(inst) - relevant
    ax.pie([relevant, other], labels=[f"Industry-\nrelevant\n{100*relevant/len(inst):.0f}%", f"Other\n{100*other/len(inst):.0f}%"],
           colors=["#4C72B0", "#CCCCCC"], startangle=90)
    ax.set_title("(c) Industry relevance")

    # (d) repositories: apache / eclipse / other org breakdown
    ax = axes[1, 1]
    repos = inst["repository"].drop_duplicates()
    n_apache = repos.str.contains("github.com:apache/").sum()
    n_eclipse = repos.str.contains("github.com:eclipse", case=False).sum()
    n_other = len(repos) - n_apache - n_eclipse
    bars = ax.bar(["Apache", "Eclipse", "Other orgs"], [n_apache, n_eclipse, n_other],
                   color=["#C44E52", "#8172B2", "#937860"])
    ax.set_title(f"(d) Repositories ({len(repos)} total)")
    ax.bar_label(bars)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "dataset_characteristics.png", dpi=300)
    plt.close(fig)
    print(f"Fig 5 -- smell counts: {counts.to_dict()}, severity: {sev_counts.to_dict()}, "
          f"industry: relevant={relevant} other={other}, repos: apache={n_apache} eclipse={n_eclipse} other={n_other} total={len(repos)}")


def fig7_algorithm_comparison():
    ml = json.load(open(ROOT / "outputs" / "ml_results.json"))
    models = ml["multiclass_4severity"]["models"]
    names = list(models.keys())
    metrics = {
        "Accuracy": [models[n]["test_accuracy"] for n in names],
        "Precision": [models[n]["test_precision_macro"] for n in names],
        "Recall": [models[n]["test_recall_macro"] for n in names],
        "F1-score": [models[n]["test_f1_macro"] for n in names],
        "ROC-AUC": [models[n]["test_roc_auc"] for n in names],
    }
    x = range(len(names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (label, vals) in enumerate(metrics.items()):
        ax.bar([xi + i * width for xi in x], vals, width, label=label)
    ax.set_xticks([xi + 2 * width for xi in x])
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Score (macro-averaged where applicable)")
    ax.set_title("ML Algorithm Comparison for Code Smell Severity Prediction (4-class)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, frameon=False)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "algorithm_comparison.png", dpi=300)
    plt.close(fig)
    print("Fig 7 written from real ml_results.json")


if __name__ == "__main__":
    fig5_dataset_characteristics()
    fig7_algorithm_comparison()
