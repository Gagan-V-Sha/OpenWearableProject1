"""
Title: 01b — Daily Dataset Cleaning (Post-merge)
Project: HCAI Open Wearables — Explainable Recovery Assistant

Description:
    Cleans the merged daily dataset produced by preprocess.py (or the committed
    combined_daily.csv). This step is intentionally lightweight: it standardizes
    types, removes rows with no usable signals, and applies simple plausibility
    bounds to reduce extreme/outlier values.

Input:
    - combined_daily.csv

Output:
    - combined_daily_clean.csv (light cleaning)
    - combined_daily_filled.csv (imputed; no empty cells in key columns)
    - combined_daily_strict.csv (drop rows with any missing key value; keeps real zeros)

Author: Shrusti
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "combined_daily.csv"
# NOTE: Excel can lock CSV files. To avoid PermissionError, we write to v2 names.
OUTPUT = ROOT / "combined_daily_clean_v2.csv"
OUTPUT_FILLED = ROOT / "combined_daily_filled_v3.csv"
OUTPUT_FILLED_FLAGS = ROOT / "combined_daily_filled_flags_v2.csv"
OUTPUT_STRICT = ROOT / "combined_daily_strict_v2.csv"


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _clip_to_range(s: pd.Series, min_val: float, max_val: float) -> pd.Series:
    s = s.copy()
    s = s.where((s >= min_val) & (s <= max_val))
    return s


def clean_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse date and standardize column types
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = _to_numeric(
        df,
        [
            "sleep_hours",
            "resting_hr",
            "steps",
            "sleep_efficiency",
            "active_minutes",
            "rmssd",
        ],
    )

    # Plausibility bounds (keep as NaN if outside range)
    if "sleep_hours" in df.columns:
        df["sleep_hours"] = _clip_to_range(df["sleep_hours"], 0.0, 24.0)
    if "sleep_efficiency" in df.columns:
        df["sleep_efficiency"] = _clip_to_range(df["sleep_efficiency"], 0.0, 100.0)
    if "resting_hr" in df.columns:
        df["resting_hr"] = _clip_to_range(df["resting_hr"], 20.0, 220.0)
    if "rmssd" in df.columns:
        df["rmssd"] = _clip_to_range(df["rmssd"], 0.0, 500.0)
    if "steps" in df.columns:
        df["steps"] = _clip_to_range(df["steps"], 0.0, 200_000.0)
    if "active_minutes" in df.columns:
        df["active_minutes"] = _clip_to_range(df["active_minutes"], 0.0, 1_440.0)

    # Drop rows that have no usable core signals (common in LifeSnaps daily export)
    core = ["sleep_hours", "resting_hr", "steps"]
    present_core = [c for c in core if c in df.columns]
    if present_core:
        df = df.loc[~df[present_core].isna().all(axis=1)].copy()

    # Remove rows with missing identifiers/dates
    df = df.dropna(subset=["user_id", "date"])

    # Sort for consistency
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values so downstream CSVs have no empty cells.

    Policy:
    - Adds `<col>_imputed` flags (1 if value was missing and got filled; else 0)
    - steps, active_minutes: fill missing with 0 (absence of recorded activity)
    - sleep_hours, resting_hr, sleep_efficiency, rmssd: fill per-user median
      (fallback to global median). Any remaining missing rows are dropped.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

    zero_fill = ["steps", "active_minutes"]
    median_fill = ["sleep_hours", "resting_hr", "sleep_efficiency", "rmssd"]

    for c in zero_fill:
        if c in df.columns:
            missing = pd.to_numeric(df[c], errors="coerce").isna()
            df[f"{c}_imputed"] = missing.astype(int)
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for c in median_fill:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        missing = df[c].isna()
        df[f"{c}_imputed"] = missing.astype(int)
        global_med = df[c].median(skipna=True)
        per_user_med = df.groupby("user_id")[c].transform("median")
        df[c] = df[c].fillna(per_user_med).fillna(global_med)

    # Final cleanup: remove anything that still has missing key fields
    key_cols = [c for c in (zero_fill + median_fill) if c in df.columns]
    df = df.dropna(subset=key_cols)

    # Restore date column to ISO date (matches combined_daily.csv style)
    df["date"] = df["date"].dt.date
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
    return df


def strict_complete_cases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows that have missing values in the key wearable columns.
    This keeps true zeros as-is and avoids any imputation/flags.
    """
    df = df.copy()
    key_cols = [
        "sleep_hours",
        "resting_hr",
        "steps",
        "sleep_efficiency",
        "active_minutes",
        "rmssd",
    ]
    key_cols = [c for c in key_cols if c in df.columns]
    df = df.dropna(subset=key_cols)
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
    return df


def main() -> None:
    df = pd.read_csv(INPUT)
    cleaned = clean_daily(df)
    cleaned.to_csv(OUTPUT, index=False)

    print(f"Wrote {OUTPUT}")
    print(f"  Rows: {len(cleaned):,} (from {len(df):,})")
    print(f"  Users: {cleaned['user_id'].nunique()}")

    filled = fill_missing(cleaned)
    filled.to_csv(OUTPUT_FILLED, index=False)

    print(f"Wrote {OUTPUT_FILLED}")
    print(f"  Rows: {len(filled):,} (from {len(cleaned):,})")
    print(f"  Users: {filled['user_id'].nunique()}")

    filled.to_csv(OUTPUT_FILLED_FLAGS, index=False)
    print(f"Wrote {OUTPUT_FILLED_FLAGS}")

    strict_df = strict_complete_cases(cleaned)
    strict_df.to_csv(OUTPUT_STRICT, index=False)
    print(f"Wrote {OUTPUT_STRICT}")
    print(f"  Rows: {len(strict_df):,} (from {len(cleaned):,})")
    print(f"  Users: {strict_df['user_id'].nunique()}")


if __name__ == "__main__":
    main()

