"""
07_feature_selection.py

The three-stage feature-selection methodology described in the paper
(mutual information filter -> Pearson correlation pruning -> RFECV),
run for real against this project's actual (compact) feature set --
11 numeric CK/context features + 2 categorical fields (smell,
industry-relevance flag), one-hot encoded for the correlation/RFECV
stages. Target: severity_consensus (4-class).

This determines whether the paper's existing methodology description
(and its Fig. 6 diagram) still applies to the final feature set used by
06_ml_classifier.py, or whether that script's "use every available
feature, prune via class-weighting instead" choice needs to be reconciled
against what selection actually recommends.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
m = importlib.import_module("06_ml_classifier")

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def main():
    df = m.build_dataset()
    y = df["severity_consensus"]

    # one-hot encode categoricals up front so every stage operates on a
    # single flat numeric feature matrix (needed for MI / correlation /
    # RFECV alike)
    num = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(df[m.NUMERIC_FEATURES]),
        columns=m.NUMERIC_FEATURES, index=df.index,
    )
    ohe = OneHotEncoder(sparse_output=False)
    cat_arr = ohe.fit_transform(df[m.CATEGORICAL_FEATURES])
    cat_cols = ohe.get_feature_names_out(m.CATEGORICAL_FEATURES)
    cat = pd.DataFrame(cat_arr, columns=cat_cols, index=df.index)
    X = pd.concat([num, cat], axis=1)

    print(f"Starting feature set ({len(X.columns)}): {list(X.columns)}")

    # --- Stage 1: mutual information filter, threshold 0.05 ---
    mi = mutual_info_classif(X, y, random_state=SEED, discrete_features=[c in cat_cols for c in X.columns])
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    print("\n--- Stage 1: mutual information (vs. severity_consensus) ---")
    print(mi_series.round(4))
    survivors_1 = mi_series[mi_series >= 0.05].index.tolist()
    dropped_1 = mi_series[mi_series < 0.05].index.tolist()
    print(f"\nDropped (MI < 0.05): {dropped_1}")
    print(f"Survive stage 1: {survivors_1}")

    # --- Stage 2: pairwise Pearson correlation pruning, |r| > 0.8 ---
    X1 = X[survivors_1]
    corr = X1.corr().abs()
    to_drop = set()
    cols = list(X1.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            if a in to_drop or b in to_drop:
                continue
            if corr.loc[a, b] > 0.8:
                # keep the one with higher stage-1 MI score
                loser = a if mi_series[a] < mi_series[b] else b
                to_drop.add(loser)
                print(f"Stage 2: {a} vs {b} r={corr.loc[a,b]:.3f} > 0.8 -- dropping {loser}")
    survivors_2 = [c for c in survivors_1 if c not in to_drop]
    print(f"\nSurvive stage 2: {survivors_2}")

    # --- Stage 3: RFECV with Random Forest ---
    X2 = X[survivors_2]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rfecv = RFECV(
        RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1),
        step=1, cv=cv, scoring="f1_macro", min_features_to_select=1, n_jobs=-1,
    )
    rfecv.fit(X2, y)
    survivors_3 = list(X2.columns[rfecv.support_])
    print(f"\n--- Stage 3: RFECV (Random Forest, f1_macro) ---")
    print(f"Optimal feature count: {rfecv.n_features_}")
    print(f"Survive stage 3 (final feature set): {survivors_3}")

    result = {
        "starting_features": list(X.columns),
        "stage1_mutual_info": mi_series.round(4).to_dict(),
        "stage1_dropped": dropped_1,
        "stage1_survivors": survivors_1,
        "stage2_dropped": sorted(to_drop),
        "stage2_survivors": survivors_2,
        "stage3_survivors": survivors_3,
        "stage3_n_features": int(rfecv.n_features_),
    }
    with open(OUT / "feature_selection_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nWrote {OUT / 'feature_selection_results.json'}")


if __name__ == "__main__":
    main()
