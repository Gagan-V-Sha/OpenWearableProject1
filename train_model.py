"""
STEP 5 — ML Module (prototype)

Trains a simple classifier to predict the rule-engine recommendation from
interpretable 7-day profile features. This is a baseline ML layer that can
later be replaced with XGBoost/LightGBM and audited for fairness.

Inputs:
  - profiles_7day.csv
  - profiles_7day_with_rules.csv (preferred; otherwise derived from 03_rule_engine.py)

Outputs:
  - model_metrics.txt (printed to stdout)
  - shap_summary.csv (per-feature mean |SHAP| on test split)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURES_NUMERIC = [
    "sleep_change_pct",
    "hr_change_bpm",
    "steps_change_pct",
    "recovery_score",
    "hr_elevation_bpm",
    "training_load_ratio",
    "current_days",
    "baseline_days",
]

FEATURES_CATEGORICAL = ["source"]

LABEL_COL = "recommendation"


def _load_training_df(root: Path) -> pd.DataFrame:
    with_rules = root / "profiles_7day_with_rules.csv"
    if with_rules.exists():
        return pd.read_csv(with_rules)

    # Fallback: compute labels using the rule engine module.
    profiles = pd.read_csv(root / "profiles_7day.csv")
    import importlib

    rule_engine = importlib.import_module("03_rule_engine")
    return rule_engine.apply_rule_engine(profiles)


def main() -> None:
    root = Path(__file__).resolve().parent
    df = _load_training_df(root)

    # Basic cleanup
    df = df.dropna(subset=FEATURES_NUMERIC + FEATURES_CATEGORICAL + [LABEL_COL]).copy()

    X = df[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", FEATURES_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CATEGORICAL),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf = Pipeline([("pre", pre), ("model", model)])
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("## Classification report (test)")
    print(classification_report(y_test, preds))
    print("## Confusion matrix (labels order = sorted unique labels)")
    labels = sorted(y.unique())
    print(labels)
    print(confusion_matrix(y_test, preds, labels=labels))

    # SHAP (tree) on numeric+encoded features for test split
    try:
        import shap

        X_test_trans = clf.named_steps["pre"].transform(X_test)
        # Get feature names
        ohe = clf.named_steps["pre"].named_transformers_["cat"]
        cat_names = list(ohe.get_feature_names_out(FEATURES_CATEGORICAL))
        feature_names = FEATURES_NUMERIC + cat_names

        explainer = shap.TreeExplainer(clf.named_steps["model"])
        shap_values = explainer.shap_values(X_test_trans)

        # SHAP shapes vary by model/version:
        # - multiclass: list of arrays [n_samples, n_features]
        # - multiclass: ndarray [n_samples, n_features, n_classes]
        # - binary/regression: ndarray [n_samples, n_features]
        if isinstance(shap_values, list):
            per_class = [np.abs(v).mean(axis=0) for v in shap_values]
            mean_abs = np.mean(np.stack(per_class, axis=0), axis=0)
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                # [n_samples, n_features, n_classes] -> mean over samples+classes
                mean_abs = np.abs(arr).mean(axis=(0, 2))
            else:
                mean_abs = np.abs(arr).mean(axis=0)

        out = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        out_path = root / "shap_summary.csv"
        out.to_csv(out_path, index=False)
        print(f"## Wrote {out_path.name}")
    except Exception as e:
        print(f"## SHAP skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

