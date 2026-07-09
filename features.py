# Feature definitions used by build_profiles.py and train_models.py.

FEATURES = [
    "sleep_change_pct",
    "hr_elevation_bpm",
    "training_load_ratio",
    "sleep_avg_7d",
    "resting_hr_avg_7d",
    "steps_avg_7d",
    "active_minutes_avg_7d",
    "rmssd_avg_7d",
    "sleep_efficiency_avg_7d",
    "workouts_count",
    "rmssd_missing",
]

# Core features that must be present for a profile row to be usable.

CORE_FEATURES = [f for f in FEATURES if f not in ("rmssd_avg_7d",)]

# Map text recommendations to integers for XGBoost
LABEL_MAP = {
    "Rest Day": 0,
    "Light Activity": 1,
    "Intensive Training": 2,
}
