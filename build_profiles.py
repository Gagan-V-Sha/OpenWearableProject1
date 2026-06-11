# Build 7-day rolling profiles and pseudo-labels for Machine Learning.

import os
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
DAILY_PATH = PROCESSED / "combined_daily.csv"
OUT_PROFILES = PROCESSED / "profiles_7day.csv"

def generate_pseudo_labels(row: pd.Series) -> tuple[float, str]:
    # Heuristic rule engine to generate a 'recovery_score' and a recommendation label.
    # Score ranges 0.0 to 1.0.
    score = 0.55
    sleep_change = row.get("sleep_change_pct", 0.0)
    hr_change = row.get("hr_elevation_bpm", 0.0)
    load = row.get("training_load_ratio", 1.0)
    
    # Missing value handling if NaNs slip through
    if pd.isna(sleep_change): sleep_change = 0.0
    if pd.isna(hr_change): hr_change = 0.0
    if pd.isna(load): load = 1.0

    # Sleep penalization
    if sleep_change < -10.0:
        score -= 0.25
    elif sleep_change < -5.0:
        score -= 0.15
    elif sleep_change > 5.0:
        score += 0.10

    # Heart rate penalization
    if hr_change > 4.0:
        score -= 0.25
    elif hr_change > 2.0:
        score -= 0.15
    elif hr_change < -2.0:
        score += 0.10

    # Training Load (ACWR) penalization
    if load > 1.50:
        score -= 0.20
    elif load > 1.30:
        score -= 0.10
    elif load < 0.85:
        score += 0.10

    score = max(0.0, min(1.0, score))

    if score < 0.45:
        rec = "Rest Day"
    elif score < 0.65:
        rec = "Light Activity"
    else:
        rec = "Intensive Training"

    return score, rec

def build_profiles() -> None:
    if not DAILY_PATH.exists():
        raise FileNotFoundError(f"Cannot find daily data at {DAILY_PATH}")
        
    print("Loading daily data...")
    df = pd.read_csv(DAILY_PATH)
    df["date"] = pd.to_datetime(df["date"])
    
    profiles = []
    
    # Process each user individually
    grouped = df.groupby("user_id")
    print(f"Processing {len(grouped)} users...")
    
    for user_id, group in grouped:
        # Sort by date
        user_df = group.sort_values("date").set_index("date").copy()
        
        # Create a complete date range to expose missing days
        idx = pd.date_range(user_df.index.min(), user_df.index.max())
        user_df = user_df.reindex(idx)
        
        # Interpolate missing values (linear, max 3 days to avoid hallucinating long gaps)
        cols_to_interpolate = ["sleep_hours", "resting_hr", "rmssd", "sleep_efficiency"]
        user_df[cols_to_interpolate] = user_df[cols_to_interpolate].interpolate(method="linear", limit=3)
        
        # Fill remaining NAs with backward/forward fill for robustness
        user_df[cols_to_interpolate] = user_df[cols_to_interpolate].bfill().ffill()
        
        # Fill activity with 0 if missing
        cols_zero = ["steps", "active_minutes"]
        user_df[cols_zero] = user_df[cols_zero].fillna(0)
        
        # 1. Calculate 7-day Rolling Features
        # Current 7 days
        roll_7d = user_df.rolling(window=7, min_periods=4)
        avg_7d = roll_7d[cols_to_interpolate].mean()
        sum_7d = roll_7d[cols_zero].sum()
        
        # Workouts count (days with > 30 active minutes)
        workouts = user_df["active_minutes"].rolling(window=7).apply(lambda x: (x > 30).sum(), raw=False)
        
        # 2. Calculate Baseline (Previous 7 days)
        # Shift the 7-day average by 7 days to get the prior week's average
        avg_prev_7d = avg_7d.shift(7)
        sum_prev_7d = sum_7d.shift(7)
        
        # 3. Derive Features
        sleep_change_pct = ((avg_7d["sleep_hours"] - avg_prev_7d["sleep_hours"]) / avg_prev_7d["sleep_hours"]) * 100
        hr_elevation_bpm = avg_7d["resting_hr"] - avg_prev_7d["resting_hr"]
        
        # Acute:Chronic Workload Ratio (ACWR proxy)
        training_load_ratio = sum_7d["active_minutes"] / sum_prev_7d["active_minutes"].replace(0, 1) # avoid div0
        
        # Build Profile Rows
        for current_date in user_df.index:
            # We need at least 14 days of history to calculate baselines properly
            if pd.isna(avg_prev_7d.loc[current_date, "sleep_hours"]):
                continue
                
            row = {
                "user_id": user_id,
                "window_end_date": current_date.strftime("%Y-%m-%d"),
                
                # Raw 7d
                "sleep_avg_7d": avg_7d.loc[current_date, "sleep_hours"],
                "resting_hr_avg_7d": avg_7d.loc[current_date, "resting_hr"],
                "rmssd_avg_7d": avg_7d.loc[current_date, "rmssd"],
                "sleep_efficiency_avg_7d": avg_7d.loc[current_date, "sleep_efficiency"],
                "steps_total_7d": sum_7d.loc[current_date, "steps"],
                "active_minutes_total_7d": sum_7d.loc[current_date, "active_minutes"],
                "workouts_count": workouts.loc[current_date],
                
                # Baseline 7d
                "sleep_avg_prev_7d": avg_prev_7d.loc[current_date, "sleep_hours"],
                "active_minutes_total_prev_7d": sum_prev_7d.loc[current_date, "active_minutes"],
                
                # Derived (ML Features)
                "sleep_change_pct": sleep_change_pct.loc[current_date],
                "hr_elevation_bpm": hr_elevation_bpm.loc[current_date],
                "training_load_ratio": training_load_ratio.loc[current_date],
            }
            
            # Generate Label
            score, rec = generate_pseudo_labels(pd.Series(row))
            row["recovery_score"] = score
            row["recommendation"] = rec
            
            profiles.append(row)

    profiles_df = pd.DataFrame(profiles)
    
    # Drop rows with any remaining NaNs to ensure clean ML training
    profiles_df = profiles_df.dropna()
    
    profiles_df.to_csv(OUT_PROFILES, index=False)
    print(f"\nWrote {OUT_PROFILES}")
    print(f"Total extracted profiles: {len(profiles_df)}")
    print("\nRecommendation distribution:")
    print(profiles_df["recommendation"].value_counts())

if __name__ == "__main__":
    build_profiles()
