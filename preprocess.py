"""Merge LifeSnaps (Fitbit) and Figshare (Samsung) into one daily wearable CSV."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = DATA / "processed" / "combined_daily.csv"

LIFESNAPS_PATH = (
    DATA / "lifesnaps" / "csv_rais_anonymized" / "daily_fitbit_sema_df_unprocessed.csv"
)
FIGSHARE_SLEEP_PATH = DATA / "figshare" / "sleep_diary.csv"
FIGSHARE_HRV_PATH = DATA / "figshare" / "sensor_hrv_filtered.csv"


def load_lifesnaps() -> pd.DataFrame:
    df = pd.read_csv(LIFESNAPS_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    active = (
        df["lightly_active_minutes"].fillna(0)
        + df["moderately_active_minutes"].fillna(0)
        + df["very_active_minutes"].fillna(0)
    )
    out = pd.DataFrame(
        {
            "user_id": "LS_" + df["id"].astype(str),
            "date": df["date"],
            "sleep_hours": df["minutesAsleep"] / 60.0,
            "resting_hr": df["resting_hr"],
            "steps": df["steps"],
            "sleep_efficiency": df["sleep_efficiency"],
            "active_minutes": active,
            "rmssd": df["rmssd"],
            "source": "lifesnaps",
        }
    )
    return out.dropna(subset=["date"])


def load_figshare() -> pd.DataFrame:
    sleep = pd.read_csv(FIGSHARE_SLEEP_PATH)
    sleep["date"] = pd.to_datetime(sleep["date"]).dt.date

    hrv = pd.read_csv(FIGSHARE_HRV_PATH)
    hrv["date"] = pd.to_datetime(hrv["ts_start"], unit="ms").dt.date
    hrv_daily = (
        hrv.groupby(["deviceId", "date"], as_index=False)
        .agg(
            resting_hr=("HR", "mean"),
            steps=("steps", "sum"),
            rmssd=("rmssd", "mean"),
            active_minutes=("steps", lambda s: (s > 0).sum() * 5),
        )
    )

    merged = sleep.merge(
        hrv_daily,
        left_on=["userId", "date"],
        right_on=["deviceId", "date"],
        how="left",
    )
    out = pd.DataFrame(
        {
            "user_id": "FS_" + merged["userId"].astype(str),
            "date": merged["date"],
            "sleep_hours": merged["sleep_duration"],
            "resting_hr": merged["resting_hr"],
            "steps": merged["steps"],
            "sleep_efficiency": merged["sleep_efficiency"] * 100,
            "active_minutes": merged["active_minutes"],
            "rmssd": merged["rmssd"],
            "source": "figshare",
        }
    )
    return out.dropna(subset=["date"])


def main() -> None:
    lifesnaps = load_lifesnaps()
    figshare = load_figshare()
    combined = pd.concat([lifesnaps, figshare], ignore_index=True)
    combined = combined.sort_values(["user_id", "date"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT, index=False)

    print(f"Wrote {OUT}")
    print(f"  Total rows: {len(combined):,}")
    print(f"  Users: {combined['user_id'].nunique()}")
    print(f"  LifeSnaps rows: {len(lifesnaps):,} ({lifesnaps['user_id'].nunique()} users)")
    print(f"  Figshare rows: {len(figshare):,} ({figshare['user_id'].nunique()} users)")
    print(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")


if __name__ == "__main__":
    main()
