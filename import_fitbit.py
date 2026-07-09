# Import your own Fitbit data into the Whyable pipeline.
#
# Two input modes:
#
#   1. Google Takeout / Fitbit account export (recommended, no OAuth needed):
#        python import_fitbit.py --takeout "path/to/Takeout/Fitbit" --user-id REAL_ME
#      Download from https://takeout.google.com (select only Fitbit) or from
#      fitbit.com account settings -> "Data Export".
#
#   2. A plain daily CSV you assembled yourself:
#        python import_fitbit.py --csv my_week.csv --user-id REAL_ME
#      Required column: date (YYYY-MM-DD). Optional columns: sleep_hours,
#      resting_hr, steps, sleep_efficiency, active_minutes, rmssd.
#
# The script maps everything into the combined_daily.csv schema, replaces any
# previous rows for the same user id, registers the user in
# user_demographics.csv, and rebuilds profiles_7day.csv so the API, model,
# SHAP explanations and DiCE suggestions all work on your real data.
#
# NOTE: you need at least ~14 consecutive days of data (7-day window + 7-day
# baseline week) before a recovery profile can be computed.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
DAILY_PATH = PROCESSED / "combined_daily.csv"
DEMO_PATH = PROCESSED / "user_demographics.csv"

DAILY_COLUMNS = [
    "user_id", "date", "sleep_hours", "resting_hr", "steps", "sleep_efficiency",
    "active_minutes", "rmssd", "sema_tired", "sema_rested", "source",
]


# --------------------------------------------------------------------------- #
# Takeout / account-export parsing
# --------------------------------------------------------------------------- #

def _parse_date(value) -> pd.Timestamp | None:
    """Fitbit exports mix formats: '2024-01-15', '01/15/24 00:00:00', ISO timestamps."""
    for fmt in ("%Y-%m-%d", "%m/%d/%y %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
        try:
            return pd.to_datetime(value, format=fmt)
        except (ValueError, TypeError):
            continue
    ts = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(ts) else ts


def _load_json_files(folder: Path, pattern: str) -> list:
    records = []
    for path in sorted(folder.rglob(pattern)):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            records.extend(data if isinstance(data, list) else [data])
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [warn] Skipping unreadable file {path.name}: {e}")
    return records


def _daily_value_series(folder: Path, pattern: str, metric: str) -> pd.Series:
    """Parse simple {dateTime, value} JSON files into a per-day series.

    Intraday files (steps) are summed per day; one-value-per-day files
    (active minutes, resting HR inner value) end up unchanged by the sum.
    """
    records = _load_json_files(folder, pattern)
    rows = []
    for r in records:
        ts = _parse_date(r.get("dateTime"))
        if ts is None:
            continue
        value = r.get("value")
        if isinstance(value, dict):  # resting_heart_rate wraps {date, value, error}
            value = value.get("value")
        try:
            rows.append((ts.normalize(), float(value)))
        except (TypeError, ValueError):
            continue
    if not rows:
        print(f"  [warn] No usable '{metric}' data found (pattern {pattern})")
        return pd.Series(dtype=float)
    s = pd.DataFrame(rows, columns=["date", "value"]).groupby("date")["value"].sum()
    print(f"  Parsed {metric}: {len(s)} days")
    return s


def _sleep_frame(folder: Path) -> pd.DataFrame:
    """sleep-*.json: list of sleep logs with dateOfSleep, minutesAsleep, efficiency."""
    records = _load_json_files(folder, "sleep-*.json")
    rows = []
    for r in records:
        ts = _parse_date(r.get("dateOfSleep"))
        if ts is None or r.get("minutesAsleep") is None:
            continue
        rows.append({
            "date": ts.normalize(),
            "minutes": float(r["minutesAsleep"]),
            "efficiency": r.get("efficiency"),
        })
    if not rows:
        print("  [warn] No usable sleep data found (pattern sleep-*.json)")
        return pd.DataFrame(columns=["sleep_hours", "sleep_efficiency"])
    df = pd.DataFrame(rows)
    # Multiple logs per night (naps): sum the minutes, keep the longest log's efficiency.
    agg = df.sort_values("minutes").groupby("date").agg(
        minutes=("minutes", "sum"), efficiency=("efficiency", "last")
    )
    out = pd.DataFrame({
        "sleep_hours": agg["minutes"] / 60.0,
        "sleep_efficiency": pd.to_numeric(agg["efficiency"], errors="coerce"),
    })
    print(f"  Parsed sleep: {len(out)} days")
    return out


def _hrv_series(folder: Path) -> pd.Series:
    """'Daily Heart Rate Variability Summary - *.csv' files with an rmssd column."""
    frames = []
    for path in sorted(folder.rglob("*Heart Rate Variability Summary*.csv")):
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as e:
            print(f"  [warn] Skipping unreadable HRV file {path.name}: {e}")
            continue
        df.columns = [c.strip().lower() for c in df.columns]
        if "rmssd" not in df.columns or "timestamp" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.normalize()
        frames.append(df.dropna(subset=["date"])[["date", "rmssd"]])
    if not frames:
        print("  [warn] No HRV (RMSSD) data found - fine if your Fitbit model "
              "doesn't measure HRV; the pipeline handles this via rmssd_missing.")
        return pd.Series(dtype=float)
    s = pd.concat(frames).groupby("date")["rmssd"].mean()
    print(f"  Parsed HRV: {len(s)} days")
    return s


def parse_takeout(folder: Path, user_id: str) -> pd.DataFrame:
    print(f"Scanning Fitbit export at {folder} ...")
    sleep = _sleep_frame(folder)
    resting_hr = _daily_value_series(folder, "resting_heart_rate-*.json", "resting heart rate")
    steps = _daily_value_series(folder, "steps-*.json", "steps")
    active = (
        _daily_value_series(folder, "lightly_active_minutes-*.json", "lightly active minutes")
        .add(_daily_value_series(folder, "moderately_active_minutes-*.json", "moderately active minutes"), fill_value=0)
        .add(_daily_value_series(folder, "very_active_minutes-*.json", "very active minutes"), fill_value=0)
    )
    rmssd = _hrv_series(folder)

    frame = pd.DataFrame({
        "sleep_hours": sleep.get("sleep_hours", pd.Series(dtype=float)),
        "sleep_efficiency": sleep.get("sleep_efficiency", pd.Series(dtype=float)),
        "resting_hr": resting_hr,
        "steps": steps,
        "active_minutes": active,
        "rmssd": rmssd,
    })
    if frame.empty:
        raise SystemExit(
            "No wearable data found. Point --takeout at the folder that contains "
            "the Fitbit export (it should hold files like sleep-*.json, "
            "steps-*.json, resting_heart_rate-*.json)."
        )

    frame = frame.sort_index()
    frame.index.name = "date"
    return _finalize(frame.reset_index(), user_id, source="fitbit_export")


# --------------------------------------------------------------------------- #
# Generic CSV parsing
# --------------------------------------------------------------------------- #

def parse_csv(path: Path, user_id: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise SystemExit("The CSV must have a 'date' column (YYYY-MM-DD).")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for col in ("sleep_hours", "resting_hr", "steps", "sleep_efficiency",
                "active_minutes", "rmssd"):
        if col not in df.columns:
            df[col] = np.nan
            print(f"  [note] Column '{col}' not in the CSV - left empty.")
    return _finalize(df, user_id, source="manual_csv")


# --------------------------------------------------------------------------- #
# Merge into the pipeline
# --------------------------------------------------------------------------- #

def _finalize(df: pd.DataFrame, user_id: str, source: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "user_id": user_id,
        "date": pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"),
        "sleep_hours": pd.to_numeric(df["sleep_hours"], errors="coerce"),
        "resting_hr": pd.to_numeric(df["resting_hr"], errors="coerce"),
        "steps": pd.to_numeric(df["steps"], errors="coerce"),
        "sleep_efficiency": pd.to_numeric(df["sleep_efficiency"], errors="coerce"),
        "active_minutes": pd.to_numeric(df["active_minutes"], errors="coerce"),
        "rmssd": pd.to_numeric(df["rmssd"], errors="coerce"),
        "sema_tired": np.nan,
        "sema_rested": np.nan,
        "source": source,
    })
    return out.sort_values("date").reset_index(drop=True)[DAILY_COLUMNS]


def merge_into_pipeline(new_rows: pd.DataFrame, user_id: str) -> None:
    if not DAILY_PATH.exists():
        raise SystemExit(f"{DAILY_PATH} not found - run preprocess.py first.")

    daily = pd.read_csv(DAILY_PATH)
    before = len(daily)
    daily = daily[daily["user_id"] != user_id]  # replace previous import
    replaced = before - len(daily)
    daily = pd.concat([daily, new_rows], ignore_index=True)
    daily = daily.sort_values(["user_id", "date"]).reset_index(drop=True)
    daily.to_csv(DAILY_PATH, index=False)
    print(f"\nWrote {len(new_rows)} days for '{user_id}' to {DAILY_PATH}"
          + (f" (replaced {replaced} previous rows)" if replaced else ""))

    if DEMO_PATH.exists():
        demo = pd.read_csv(DEMO_PATH)
        if user_id not in set(demo["user_id"]):
            demo = pd.concat([demo, pd.DataFrame([{
                "gender": "Unknown", "user_id": user_id,
                "sport_type": "Mixed", "age_group": "Unknown",
            }])], ignore_index=True)
            demo.to_csv(DEMO_PATH, index=False)
            print(f"Registered '{user_id}' in {DEMO_PATH}")


def main():
    ap = argparse.ArgumentParser(description="Import personal Fitbit data into Whyable.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--takeout", type=Path, help="Path to the Fitbit folder of a Google Takeout export")
    src.add_argument("--csv", type=Path, help="Path to a daily CSV (date, sleep_hours, resting_hr, ...)")
    ap.add_argument("--user-id", default="REAL_ME", help="User id for the dashboard (default REAL_ME)")
    ap.add_argument("--no-rebuild", action="store_true", help="Skip rebuilding profiles_7day.csv")
    args = ap.parse_args()

    if args.takeout:
        rows = parse_takeout(args.takeout, args.user_id)
    else:
        rows = parse_csv(args.csv, args.user_id)

    n_days = rows["date"].nunique()
    print(f"\nImported {n_days} days ({rows['date'].min()} to {rows['date'].max()})")
    if n_days < 14:
        print("[warn] Fewer than 14 days - the 7-day window + 7-day baseline may "
              "not be computable, so this user may not get a recovery profile.")

    merge_into_pipeline(rows, args.user_id)

    if not args.no_rebuild:
        print("\nRebuilding 7-day profiles (this takes a moment)...")
        from build_profiles import build_profiles
        build_profiles()
        print(f"\nDone. Start the API (uvicorn api:app --port 8000) and pick "
              f"'{args.user_id}' in the dashboard dropdown.")


if __name__ == "__main__":
    main()
