

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

CORE_FEATURES = [f for f in FEATURES if f not in ("rmssd_avg_7d",)]

LABEL_MAP = {
    "Rest Day": 0,
    "Light Activity": 1,
    "Intensive Training": 2,
}
