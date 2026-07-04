# Train the XGBoost classifier and the Isolation Forest for anomaly detection.

from pathlib import Path
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from features import FEATURES, LABEL_MAP

TARGET = "rule_recommendation"

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MODELS_DIR = ROOT / "models"
PROFILE_PATH = PROCESSED / "profiles_7day.csv"


def train_models():
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(f"Cannot find profile data at {PROFILE_PATH}")

    print("Loading profiles data...")
    df = pd.read_csv(PROFILE_PATH)

    # rmssd_avg_7d may be NaN, XGBoost handles it.
    labeled = df.dropna(subset=[TARGET]).copy()
    labeled = labeled.dropna(subset=[f for f in FEATURES if f != "rmssd_avg_7d"])
    print(f"Training target: {TARGET!r}")
    print(f"Labeled profiles: {len(labeled)} from {labeled['user_id'].nunique()} users")

    X = labeled[FEATURES]
    y = labeled[TARGET].map(LABEL_MAP)
    groups = labeled["user_id"]

    # Split BY USER so no user's overlapping 7-day windows appear in both train and test (Here there was mix which is why the accuracy was so high!).
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Train: {len(X_train)} rows / {groups.iloc[train_idx].nunique()} users | "
          f"Test: {len(X_test)} rows / {groups.iloc[test_idx].nunique()} users")
    assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx])), \
        "User leakage between train and test!"

    # Class imbalance handled with balanced sample weights.
    label_names = {v: k for k, v in LABEL_MAP.items()}
    print("Train class counts:",
          {label_names[k]: int(v) for k, v in zip(*np.unique(y_train, return_counts=True))})
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    # Fairness feedback loop: fold in reweighing factors emitted by a flagged fairness audit
    weights_path = PROCESSED / "fairness_weights.csv"
    if weights_path.exists():
        fw = pd.read_csv(weights_path)
        key = labeled.iloc[train_idx][["user_id", "window_end_date"]].merge(
            fw, on=["user_id", "window_end_date"], how="left")
        factors = key["fairness_weight"].fillna(1.0).to_numpy()
        sample_weight = sample_weight * factors
        print(f"Applied fairness reweighing from {weights_path.name} "
              f"(factor range {factors.min():.2f}-{factors.max():.2f}).")

    print("\nTraining XGBoost Classifier")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="mlogloss",
    )
    xgb_model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    metric_name = ("Surrogate fidelity to rule engine"
                   if TARGET == "rule_recommendation" else "Predictive accuracy vs SEMA")
    print(f"\n{metric_name} (held-out users): {acc:.2%}")
    print("\nClassification Report:")
    present = sorted(set(y_test) | set(y_pred))
    print(classification_report(
        y_test, y_pred,
        labels=present,
        target_names=[label_names[i] for i in present],
        zero_division=0,
    ))

    # Secondary metric: agreement with human self-reports on users.
    if TARGET == "rule_recommendation" and "recommendation" in labeled.columns:
        sema = labeled.iloc[test_idx].dropna(subset=["recommendation"])
        if len(sema):
            sema_pred = xgb_model.predict(sema[FEATURES])
            sema_true = sema["recommendation"].map(LABEL_MAP)
            print(f"[Reality check] Agreement with SEMA self-reports on held-out "
                  f"users: {accuracy_score(sema_true, sema_pred):.2%} "
                  f"({len(sema)} labeled days) - reported as a secondary finding, "
                  f"not the headline metric.")

    # Isolation Forest — on training not on test.
    print("\nTraining Isolation Forest (Anomaly Detection)")
    iso_features = [f for f in FEATURES if f != "rmssd_avg_7d"]
    iso_model = IsolationForest(contamination=0.05, random_state=42)
    iso_model.fit(X_train[iso_features])
    anomalies = iso_model.predict(X_test[iso_features])
    n_anom = int((anomalies == -1).sum())
    print(f"Isolation Forest flagged {n_anom}/{len(X_test)} held-out days "
          f"as anomalies ({n_anom / len(X_test):.1%}).")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    xgb_model.save_model(str(MODELS_DIR / "xgboost_recovery.json"))
    with open(MODELS_DIR / "isolation_forest.pkl", "wb") as f:
        pickle.dump({"model": iso_model, "features": iso_features}, f)
    print(f"\nModels successfully saved to {MODELS_DIR}/")


if __name__ == "__main__":
    train_models()
