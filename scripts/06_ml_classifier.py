"""
06_ml_classifier.py

O4: can a model trained on code measures and reviewer context predict
severity with useful reliability? Real training/evaluation on this
project's own dataset (never done here before this script -- the
draft paper's prior "Comparative Model Performance" numbers were not
produced by this project; see METHOD.md).

Features (complete-case on the merge of instances.csv + complexity.csv,
i.e. instances with a successful CK join -- 85.5% of the full set):
  - smell (one-hot: blob, data class, feature envy, long method) -- this
    also implicitly encodes function-vs-class, so `type` is dropped as
    redundant.
  - is_from_industry_relevant_project_raw (one-hot: '0', '0,5', '1')
  - n_reviewers
  - structural metrics from CK: cbo, rfc, loc are populated for BOTH
    class and function smells; wmc, dit, noc, lcom, total_methods,
    total_fields are class-smell-only and method_cyclomatic_complexity is
    function-smell-only (structurally NULL for the other type, not
    missing data in the "something went wrong" sense). Median-imputed
    within a sklearn Pipeline for model training ONLY -- this is a
    standard, transparent ML modeling step, not a reported empirical
    value, and is documented here rather than silently done.

Two targets, both from severity_consensus: the full 4-class problem, and
the binary major+critical-vs-rest split (the practically useful framing
given how rare severe cases are). Macro-averaged metrics throughout --
with 90%% "none", accuracy alone is not a meaningful headline number here.

5 algorithms (3 ensemble + 2 non-ensemble, matching the draft's existing
"three ensemble methods outperform Logistic Regression and SVM" framing,
now evaluated for real): Random Forest, Gradient Boosting, Extra Trees,
Logistic Regression, SVM. class_weight="balanced" wherever supported.
Stratified 80/20 train/test split, 5-fold stratified CV on the training
set, real confusion matrices, and a CV-vs-test comparison (the draft's
"Generalisation" check) -- all seeded (SEED=42) and written to
outputs/ml_results.json plus outputs/figures/*.png so nothing is
hand-typed into the paper.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve,
)

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEVERITY_ORDER = ["none", "minor", "major", "critical"]
NUMERIC_FEATURES = ["cbo", "rfc", "loc", "wmc", "dit", "noc", "lcom",
                     "total_methods", "total_fields", "method_cyclomatic_complexity", "n_reviewers"]
CATEGORICAL_FEATURES = ["smell", "is_from_industry_relevant_project_raw"]


def build_dataset():
    inst = pd.read_csv(ROOT / "data" / "instances.csv")
    complexity = pd.read_csv(ROOT / "data" / "complexity.csv")
    df = inst.merge(
        complexity[["instance_id"] + [c for c in NUMERIC_FEATURES if c != "n_reviewers"]],
        on="instance_id", how="inner",
    )
    print(f"Complete-case ML dataset: {len(df)} / {len(inst)} instances "
          f"({100*len(df)/len(inst):.1f}%) with a successful complexity join.")
    df["is_from_industry_relevant_project_raw"] = df["is_from_industry_relevant_project_raw"].astype(str)
    df["severe_binary"] = df["severity_consensus"].isin(["major", "critical"]).map({True: "severe", False: "not_severe"})
    return df


def make_pipeline(model):
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def get_models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=SEED),
    }


def evaluate_task(df, target_col, labels, task_name, results):
    print("\n" + "=" * 70)
    print(f"TASK: {task_name} (target = {target_col}, classes = {labels})")
    print("=" * 70)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED,
    )
    print(f"Train N={len(X_train)}, Test N={len(X_test)}")
    print("Train class distribution:", y_train.value_counts().to_dict())
    print("Test class distribution:", y_test.value_counts().to_dict())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    task_results = {"n_train": len(X_train), "n_test": len(X_test),
                     "train_dist": y_train.value_counts().to_dict(),
                     "test_dist": y_test.value_counts().to_dict(), "models": {}}

    for name, model in get_models().items():
        pipe = make_pipeline(model)
        cv_scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["f1_macro", "accuracy"], n_jobs=1)
        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_seconds = time.time() - t0
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        auc = None
        try:
            y_proba = pipe.predict_proba(X_test)
            if len(labels) == 2:
                auc = roc_auc_score((y_test == labels[1]).astype(int), y_proba[:, list(pipe.classes_).index(labels[1])])
            else:
                auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro", labels=pipe.classes_)
        except Exception as e:
            print(f"  ({name}: ROC-AUC unavailable: {e})")

        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{task_name} -- {name}")
        fname = FIG_DIR / f"confusion_{task_name.replace(' ', '_')}_{name.replace(' ', '_')}.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"\n{name}: CV f1_macro={cv_scores['test_f1_macro'].mean():.4f} "
              f"(+/-{cv_scores['test_f1_macro'].std():.4f}), "
              f"test f1_macro={f1:.4f}, test acc={acc:.4f}, "
              f"test precision={prec:.4f}, test recall={rec:.4f}, "
              f"AUC={auc if auc is None else round(auc,4)}, "
              f"train_time={train_seconds:.2f}s")

        task_results["models"][name] = {
            "cv_f1_macro_mean": float(cv_scores["test_f1_macro"].mean()),
            "cv_f1_macro_std": float(cv_scores["test_f1_macro"].std()),
            "cv_accuracy_mean": float(cv_scores["test_accuracy"].mean()),
            "test_accuracy": float(acc), "test_precision_macro": float(prec),
            "test_recall_macro": float(rec), "test_f1_macro": float(f1),
            "test_roc_auc": None if auc is None else float(auc),
            "train_seconds": float(train_seconds),
            "confusion_matrix": cm.tolist(), "confusion_matrix_labels": labels,
        }

    best = max(task_results["models"].items(), key=lambda kv: kv[1]["test_f1_macro"])
    print(f"\nBest model by test macro-F1: {best[0]} (F1={best[1]['test_f1_macro']:.4f})")
    task_results["best_model"] = best[0]

    # CV-vs-test bar chart (the draft's "Generalisation" check)
    names = list(task_results["models"].keys())
    cv_f1 = [task_results["models"][n]["cv_f1_macro_mean"] for n in names]
    test_f1 = [task_results["models"][n]["test_f1_macro"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, cv_f1, 0.4, label="CV (train) macro-F1")
    ax.bar(x + 0.2, test_f1, 0.4, label="Held-out test macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Macro-F1")
    ax.set_title(f"CV vs. test macro-F1 -- {task_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"cv_vs_test_{task_name.replace(' ', '_')}.png", dpi=300)
    plt.close(fig)

    results[task_name] = task_results


def plot_multiclass_roc(df, results):
    """One-vs-rest ROC curves for the Random Forest multiclass model (the
    model this paper's prose narrative discusses throughout, for
    consistency, even though Extra Trees scores marginally higher on
    macro-F1 -- see the comparison table for the full picture). Real
    per-class AUCs, not a boundary-pairwise metric the draft's prior text
    implied but that was never computed."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["severity_consensus"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pipe = make_pipeline(RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1))
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)
    classes = list(pipe.classes_)

    fig, ax = plt.subplots(figsize=(6, 6))
    per_class_auc = {}
    for i, c in enumerate(classes):
        y_bin = (y_test == c).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, proba[:, i])
        auc = roc_auc_score(y_bin, proba[:, i])
        per_class_auc[c] = float(auc)
        ax.plot(fpr, tpr, label=f"{c} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC -- Random Forest, 4-class severity")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_multiclass_4severity_Random_Forest.png", dpi=300)
    plt.close(fig)
    print("\nPer-class one-vs-rest AUC (Random Forest, multiclass):", per_class_auc)
    results["multiclass_4severity"]["random_forest_per_class_auc"] = per_class_auc


def main():
    df = build_dataset()
    results = {"n_complete_case": len(df), "seed": SEED}
    evaluate_task(df, "severity_consensus", SEVERITY_ORDER, "multiclass_4severity", results)
    evaluate_task(df, "severe_binary", ["not_severe", "severe"], "binary_majorcritical", results)
    plot_multiclass_roc(df, results)

    with open(OUT / "ml_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nWrote {OUT / 'ml_results.json'} and figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
