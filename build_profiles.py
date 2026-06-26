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
    # Weighted multi-signal heuristic grounded in published sports science.

    score = 0.50

    sleep_change = row.get("sleep_change_pct", 0.0)
    hr_change = row.get("hr_elevation_bpm", 0.0)
    load = row.get("training_load_ratio", 1.0)
    rmssd = row.get("rmssd_avg_7d", 40.0)
    efficiency = row.get("sleep_efficiency_avg_7d", 80.0)
    workouts = row.get("workouts_count", 3.0)

    # Missing value handling
    if pd.isna(sleep_change): sleep_change = 0.0
    if pd.isna(hr_change): hr_change = 0.0
    if pd.isna(load): load = 1.0
    if pd.isna(rmssd): rmssd = 40.0
    if pd.isna(efficiency): efficiency = 80.0
    if pd.isna(workouts): workouts = 3.0

    # Signal 1: HRV / RMSSD (Plews et al. 2013)
    # Highest weight — suppressed HRV is the strongest indicator of under-recovery.
    if rmssd > 60.0:
        score += 0.15    # Excellent autonomic recovery
    elif rmssd >= 40.0:
        score += 0.05    # Good
    elif rmssd >= 25.0:
        score -= 0.10    # Suppressed HRV
    else:
        score -= 0.20    # Very suppressed — strong rest signal

    # Signal 2: Sleep change % (Fullagar et al. 2015)
    if sleep_change > 5.0:
        score += 0.08    # More sleep than baseline = extra recovery
    elif sleep_change < -10.0:
        score -= 0.20    # Major sleep deficit
    elif sleep_change < -5.0:
        score -= 0.12    # Moderate sleep deficit

    # Signal 3: Resting HR elevation
    if hr_change < -2.0:
        score += 0.08    # Lowering HR = improved cardiovascular fitness
    elif hr_change <= 2.0:
        pass             # Neutral range
    elif hr_change <= 4.0:
        score -= 0.10    # Elevated — possible stress or fatigue
    else:
        score -= 0.18    # Significantly elevated — overreaching marker

    # Signal 4: ACWR Training Load (Gabbett 2016 published zones)
    if load < 0.8:
        score -= 0.05    # Undertraining — slight deconditioning risk
    elif load <= 1.3:
        score += 0.10    # Sweet spot — optimal training stimulus
    elif load <= 1.5:
        score -= 0.08    # Caution zone — elevated injury risk
    else:
        score -= 0.20    # Danger zone — high injury risk

    # Signal 5: Sleep efficiency
    if efficiency > 85.0:
        score += 0.07    # Clinically good sleep quality
    elif efficiency < 75.0:
        score -= 0.10    # Clinically poor sleep quality

    # Signal 6: Workouts count in last 7 days 
    if workouts > 5:
        score -= 0.08    # No rest days — overtraining risk
    elif workouts >= 3:
        score += 0.05    # Appropriate training frequency

    # Clamp to valid range
    score = max(0.0, min(1.0, score))

    # Add small Gaussian noise to prevent the model from memorising a deterministic decision boundary. Simulates real-world annotation uncertainty at class margins (σ=0.03).
    noise = np.random.normal(0.0, 0.03)
    score_noisy = max(0.0, min(1.0, score + noise))

    # Label thresholds
    if score_noisy < 0.40:
        rec = "Rest Day"
    elif score_noisy < 0.60:
        rec = "Light Activity"
    else:
        rec = "Intensive Training"

    # Return the clean score
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
        
        # Interpolate missing values.
        # Sleep and HR use limit=7 to bridge moderate gaps (up to 1 week) and recover users with small data holes. RMSSD uses limit=3 — HRV values should not be fabricated across long gaps as they carry high clinical weight.
        cols_sleep_hr = ["sleep_hours", "resting_hr", "sleep_efficiency"]
        cols_hrv = ["rmssd"]
        user_df[cols_sleep_hr] = user_df[cols_sleep_hr].interpolate(method="linear", limit=7)
        user_df[cols_hrv] = user_df[cols_hrv].interpolate(method="linear", limit=3)
        cols_to_interpolate = cols_sleep_hr + cols_hrv
        
        # Fill remaining NAs with backward/forward fill for robustness
        user_df[cols_to_interpolate] = user_df[cols_to_interpolate].bfill().ffill()
        
        # Fill activity with 0 if missing
        cols_zero = ["steps", "active_minutes"]
        user_df[cols_zero] = user_df[cols_zero].fillna(0)
        
        # RMSSD may be entirely absent for some users (device limitation).
        # Fill with 0 and treats 0 RMSSD as a strong rest signal (score -= 0.20), which is
        user_df["rmssd"] = user_df["rmssd"].fillna(0)
        
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
        
        # Acute:Chronic Workload Ratio 
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
                
                # Derived ML Features
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
    
    # Drop rows with NaNs only in the ML feature columns used for training.
    # This preserves users who have no RMSSD data (rmssd_avg_7d=0 is valid).
    from train_models import FEATURES
    profiles_df = profiles_df.dropna(subset=FEATURES + ["recommendation"])
    
    profiles_df.to_csv(OUT_PROFILES, index=False)
    print(f"\nWrote {OUT_PROFILES}")
    print(f"Total extracted profiles: {len(profiles_df)}")
    print("\nRecommendation distribution:")
    print(profiles_df["recommendation"].value_counts())

if __name__ == "__main__":
    build_profiles()
