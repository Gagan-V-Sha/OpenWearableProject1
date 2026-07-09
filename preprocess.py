# Preprocess LifeSnaps daily wearable data and extract demographics.

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"

LIFESNAPS_PATH = (
    DATA / "lifesnaps" / "rais_anonymized" / "csv_rais_anonymized" / "daily_fitbit_sema_df_unprocessed.csv"
)
OUT_DAILY = PROCESSED / "combined_daily.csv"
OUT_DEMO = PROCESSED / "user_demographics.csv"

def process_lifesnaps() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not LIFESNAPS_PATH.exists():
        raise FileNotFoundError(f"Cannot find LifeSnaps data at {LIFESNAPS_PATH}")
        
    print("Loading LifeSnaps data...")
    df = pd.read_csv(LIFESNAPS_PATH)
    
    # Filter out rows missing date
    df = df.dropna(subset=["date"]).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    
    # 1. Process Daily Wearable Data
    # min_count=1 keeps the sum NaN when ALL three components are missing (device not worn / not synced) instead of silently producing 0.
    active = df[
        ["lightly_active_minutes", "moderately_active_minutes", "very_active_minutes"]
    ].sum(axis=1, min_count=1)
    
    daily_df = pd.DataFrame({
        "user_id": "LS_" + df["id"].astype(str),
        "date": df["date"],
        "sleep_hours": df["minutesAsleep"] / 60.0,
        "resting_hr": df["resting_hr"],
        "steps": df["steps"],
        "sleep_efficiency": df["sleep_efficiency"],
        "active_minutes": active,
        "rmssd": df["rmssd"],  # Heart Rate Variability
        # SEMA ecological momentary assessment self-reports (binary, sparse).
        # These provide human ground-truth labels for the ML module, so the model is not trained on labels derived from its own input features.(which was done before and hence we got 99+ accuracy-dont forget this!!!!!!!!!!!!!)
        "sema_tired": df["TIRED"],
        "sema_rested": df["RESTED/RELAXED"],
        "source": "lifesnaps",
    })
    
    daily_df = daily_df.sort_values(["user_id", "date"]).reset_index(drop=True)
    
    # 2. Extract User Demographics
    # Taking the most common or first non-null demographic value for each user
    demo_cols = ["id", "age", "gender"]
    demo_available = [c for c in demo_cols if c in df.columns]
    
    if len(demo_available) > 1:
        # Group by id and take the first valid mode
        demo_df = df[demo_available].groupby("id").agg(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
        demo_df["user_id"] = "LS_" + demo_df["id"].astype(str)
        demo_df = demo_df.drop(columns=["id"])
        
        # Add a default sport_type for the fairness audit to work
        demo_df["sport_type"] = "Mixed" 
        
        # Rename or clean up age&gender if needed
        # Assuming gender is string. We might want to standardize it.
        if "gender" in demo_df.columns:
            demo_df["gender"] = demo_df["gender"].fillna("Unknown")
            
        if "age" in demo_df.columns:
            demo_df["age_group"] = demo_df["age"].fillna("Unknown")
            demo_df = demo_df.drop(columns=["age"])
    else:
        demo_df = pd.DataFrame(columns=["user_id", "gender", "age_group", "sport_type"])
        
    return daily_df, demo_df

def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    
    daily_df, demo_df = process_lifesnaps()
    
    # Save Daily Data
    daily_df.to_csv(OUT_DAILY, index=False)
    print(f"Wrote {OUT_DAILY}")
    print(f"  Total daily rows: {len(daily_df):,}")
    print(f"  Total users: {daily_df['user_id'].nunique()}")
    print(f"  Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    
    # Save Demographics
    if not demo_df.empty:
        demo_df.to_csv(OUT_DEMO, index=False)
        print(f"\nWrote {OUT_DEMO}")
        print(f"  Total demographic profiles: {len(demo_df)}")
        print("  Gender split:")
        if "gender" in demo_df.columns:
            print(demo_df["gender"].value_counts().to_string())

if __name__ == "__main__":
    main()
