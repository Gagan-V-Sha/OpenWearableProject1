# Train Machine Learning Models for the Open Wearables Project.
#
# Trains:
# 1. XGBoost Classifier (for recovery recommendations, compatible with SHAP)
# 2. Isolation Forest (for biological anomaly detection)

import os
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MODELS_DIR = ROOT / "models"

PROFILE_PATH = PROCESSED / "profiles_7day.csv"

# The features we want the model to learn from
FEATURES = [
    'sleep_change_pct',
    'hr_elevation_bpm',
    'training_load_ratio',
    'sleep_avg_7d',
    'resting_hr_avg_7d',
    'steps_total_7d',
    'active_minutes_total_7d',
    'rmssd_avg_7d',
    'sleep_efficiency_avg_7d',
    'workouts_count'
]

# Map text recommendations to integers for XGBoost
LABEL_MAP = {
    "Rest Day": 0,
    "Light Activity": 1,
    "Intensive Training": 2
}

def train_models():
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(f"Cannot find profile data at {PROFILE_PATH}")
        
    print("Loading profiles data...")
    df = pd.read_csv(PROFILE_PATH)
    
    # 1. Prepare Data
    # Drop rows with NaNs in feature columns just in case
    df = df.dropna(subset=FEATURES + ['recommendation'])
    
    X = df[FEATURES]
    y = df['recommendation'].map(LABEL_MAP)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    
    # 2. Train XGBoost Classifier
    print("\n--- Training XGBoost Classifier ---")
    # Using specific hyperparams to prevent overfitting and make rules cleaner for SHAP
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    xgb_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"XGBoost Accuracy: {acc:.2%}")
    print("\nClassification Report:")
    target_names = {v: k for k, v in LABEL_MAP.items()}
    print(classification_report(y_test, y_pred, target_names=[target_names[i] for i in range(3)]))
    
    # 3. Train Isolation Forest (Anomaly Detection)
    print("\n--- Training Isolation Forest (Anomaly Detection) ---")
    # Contamination set to 5% assuming 5% of days might be wild physiological outliers
    iso_model = IsolationForest(contamination=0.05, random_state=42)
    iso_model.fit(X)  # Unsupervised, so we train on all X
    
    # Let's see how many anomalies it found
    anomalies = iso_model.predict(X)
    n_anomalies = list(anomalies).count(-1)
    print(f"Isolation Forest flagged {n_anomalies} out of {len(X)} days as anomalies ({(n_anomalies/len(X)):.1%}).")
    
    # 4. Save Models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    xgb_path = MODELS_DIR / "xgboost_recovery.json"
    iso_path = MODELS_DIR / "isolation_forest.pkl"
    
    # XGBoost can save directly to JSON
    xgb_model.save_model(str(xgb_path))
    
    # Isolation forest uses pickle
    with open(iso_path, "wb") as f:
        pickle.dump(iso_model, f)
        
    print(f"\nModels successfully saved to {MODELS_DIR}/")

if __name__ == "__main__":
    train_models()
