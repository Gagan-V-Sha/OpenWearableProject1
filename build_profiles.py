# Build 7-day rolling profiles for Machine Learning.

from pathlib import Path
import pandas as pd
import numpy as np

from features import CORE_FEATURES
from rule_engine import assess as rule_assess

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
DAILY_PATH = PROCESSED / "combined_daily.csv"
OUT_PROFILES = PROCESSED / "profiles_7day.csv"
# A 7-day window needs at least this many real days.
MIN_DAYS_PER_WINDOW = 4
# Mean daily active minutes below this is treated as no meaningful load.
MIN_CHRONIC_LOAD = 10.0

def sema_label(tired, rested) -> str | float:
    if pd.isna(tired) and pd.isna(rested):
        return np.nan
    tired = 0.0 if pd.isna(tired) else tired
    rested = 0.0 if pd.isna(rested) else rested
    if tired == 1.0 and rested == 0.0:
        return "Rest Day"
    if rested == 1.0 and tired == 0.0:
        return "Intensive Training"
    return "Light Activity" # neutral self-report

def build_user_profiles(user_id: str, group: pd.DataFrame) -> pd.DataFrame:
    user_df = group.sort_values("date").set_index("date").copy()
# Complete date range to expose missing days
    idx = pd.date_range(user_df.index.min(), user_df.index.max())
    user_df = user_df.reindex(idx)
 # Interpolate inside gaps only, with strict limits: long gaps stay NaN and the d is dropped.
    cols_sleep_hr = ["sleep_hours", "resting_hr", "sleep_efficiency"]
    user_df[cols_sleep_hr] = user_df[cols_sleep_hr].interpolate(
        method="linear", limit=7, limit_area="inside"
    )
    user_df["rmssd"] = user_df["rmssd"].interpolate(
        method="linear", limit=3, limit_area="inside"
    )
  # Device-not-worn days become NaN so rolling stats skip them,
    # instead of counting fake "0 activity" days.
    worn = (user_df["steps"].fillna(0) > 0) | (user_df["active_minutes"].fillna(0) > 0)
    user_df.loc[~worn, ["steps", "active_minutes"]] = np.nan
# 7-day rolling stats over available days (>= MIN_DAYS_PER_WINDOW real days)
    roll = user_df.rolling(window=7, min_periods=MIN_DAYS_PER_WINDOW)
    sleep_avg = roll["sleep_hours"].mean()
    hr_avg = roll["resting_hr"].mean()
    eff_avg = roll["sleep_efficiency"].mean()
    rmssd_avg = roll["rmssd"].mean()
    steps_avg = roll["steps"].mean()
    active_avg = roll["active_minutes"].mean()
 # Workouts: worn days with > 30 active minutes
    is_workout = user_df["active_minutes"].gt(30.0).astype(float)
    is_workout[user_df["active_minutes"].isna()] = np.nan
    workouts = is_workout.rolling(window=7, min_periods=MIN_DAYS_PER_WINDOW).sum()
 # Baselines: previous week's rolling stats
    sleep_prev = sleep_avg.shift(7)
    hr_prev = hr_avg.shift(7)
    active_prev = active_avg.shift(7)

    sleep_change_pct = (sleep_avg - sleep_prev) / sleep_prev * 100.0

    hr_elevation_bpm = hr_avg - hr_prev

    load_ratio = active_avg / active_prev
    both_negligible = (active_avg < MIN_CHRONIC_LOAD) & (active_prev < MIN_CHRONIC_LOAD)
    load_ratio[both_negligible] = 1.0
    load_ratio[(active_prev < MIN_CHRONIC_LOAD) & ~both_negligible] = np.nan
    load_ratio = load_ratio.clip(upper=5.0)

    prof = pd.DataFrame({
        "user_id": user_id,
        "window_end_date": user_df.index.strftime("%Y-%m-%d"),

        "sleep_avg_7d": sleep_avg,
        "resting_hr_avg_7d": hr_avg,
        "rmssd_avg_7d": rmssd_avg,
        "sleep_efficiency_avg_7d": eff_avg,
        "steps_avg_7d": steps_avg,
        "active_minutes_avg_7d": active_avg,
        "workouts_count": workouts,

        "sleep_avg_prev_7d": sleep_prev,
        "active_minutes_avg_prev_7d": active_prev,

        "sleep_change_pct": sleep_change_pct,
        "hr_elevation_bpm": hr_elevation_bpm,
        "training_load_ratio": load_ratio,
    }, index=user_df.index)

    prof["rmssd_missing"] = prof["rmssd_avg_7d"].isna().astype(int)

    prof["recommendation"] = [
        sema_label(t, r) for t, r in zip(user_df["sema_tired"], user_df["sema_rested"])
    ]

    return prof

def build_profiles() -> None:
    if not DAILY_PATH.exists():
        raise FileNotFoundError(f"Cannot find daily data at {DAILY_PATH}")

    print("Loading daily data...")
    df = pd.read_csv(DAILY_PATH)
    df["date"] = pd.to_datetime(df["date"])

    grouped = df.groupby("user_id")
    print(f"Processing {len(grouped)} users...")

    profiles_df = pd.concat(
        [build_user_profiles(uid, g) for uid, g in grouped], ignore_index=True
    )
 # Keep only windows where all core features aRE from real data.
    n_before = len(profiles_df)
    profiles_df = profiles_df.dropna(subset=CORE_FEATURES).reset_index(drop=True)
    print(f"Dropped {n_before - len(profiles_df)} windows with insufficient real data.")

    rule_out = [rule_assess(r) for _, r in profiles_df.iterrows()]
    profiles_df["rule_score"] = [a.score for a in rule_out]
    profiles_df["rule_recommendation"] = [a.recommendation for a in rule_out]

    profiles_df.to_csv(OUT_PROFILES, index=False)
    print(f"\nWrote {OUT_PROFILES}")
    print(f"Total profiles: {len(profiles_df)}")
    n_labeled = profiles_df["recommendation"].notna().sum()
    print(f"Profiles with SEMA ground-truth label: {n_labeled} "
          f"({n_labeled / len(profiles_df):.1%})")
    print(f"Profiles with missing HRV: {profiles_df['rmssd_missing'].sum()}")
    print("\nSEMA label distribution (ML target):")
    print(profiles_df["recommendation"].value_counts())
    print("\nRule engine recommendation distribution (transparent layer):")
    print(profiles_df["rule_recommendation"].value_counts())

if __name__ == "__main__":
    build_profiles()
