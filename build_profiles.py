"""
Title: 02 — 7-Day Profile Builder & Recovery Features
Project: HCAI Open Wearables — Explainable Recovery Assistant

Description:
    Builds rolling 7-day user profiles and compares each window to the
    previous 7-day baseline (sleep, heart rate, steps, activity load).

Input:
    - combined_daily.csv

Output:
    - profiles_7day.csv (recovery_score, hr_elevation_bpm, training_load_ratio)

Author: Shrusti
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "combined_daily.csv"
OUTPUT = ROOT / "profiles_7day.csv"

WINDOW_DAYS = 7
MIN_DAYS_PER_WINDOW = 4


def _window_stats(group: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict | None:
    window = group[(group["date"] > start) & (group["date"] <= end)]
    if len(window) < MIN_DAYS_PER_WINDOW:
        return None
    return {
        "days": len(window),
        "sleep_avg": window["sleep_hours"].mean(),
        "resting_hr_avg": window["resting_hr"].mean(),
        "steps_total": window["steps"].sum(),
        "active_minutes_total": window["active_minutes"].sum(),
    }


def _recovery_score(sleep_change_pct: float, hr_change_bpm: float, load_ratio: float) -> float:
    score = 0.55
    if sleep_change_pct < -0.05:
        score -= 0.20
    elif sleep_change_pct > 0.05:
        score += 0.10
    if hr_change_bpm > 3:
        score -= 0.20
    elif hr_change_bpm < -2:
        score += 0.10
    if load_ratio > 1.30:
        score -= 0.15
    elif load_ratio < 0.85:
        score += 0.05
    return float(np.clip(score, 0.0, 1.0))


def build_profiles(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    rows: list[dict] = []

    for user_id, group in daily.groupby("user_id"):
        group = group.sort_values("date").reset_index(drop=True)
        source = group["source"].iloc[0]
        dates = group["date"].sort_values().unique()

        for end_date in dates:
            current_start = end_date - pd.Timedelta(days=WINDOW_DAYS)
            baseline_start = end_date - pd.Timedelta(days=2 * WINDOW_DAYS)
            baseline_end = end_date - pd.Timedelta(days=WINDOW_DAYS)

            current = _window_stats(group, current_start, end_date)
            baseline = _window_stats(group, baseline_start, baseline_end)
            if current is None or baseline is None:
                continue

            sleep_change_pct = (
                (current["sleep_avg"] - baseline["sleep_avg"]) / baseline["sleep_avg"]
                if baseline["sleep_avg"] > 0
                else 0.0
            )
            hr_change_bpm = current["resting_hr_avg"] - baseline["resting_hr_avg"]
            steps_change_pct = (
                (current["steps_total"] - baseline["steps_total"]) / baseline["steps_total"]
                if baseline["steps_total"] > 0
                else 0.0
            )
            training_load_ratio = (
                current["active_minutes_total"] / baseline["active_minutes_total"]
                if baseline["active_minutes_total"] > 0
                else 1.0
            )

            rows.append(
                {
                    "user_id": user_id,
                    "window_end_date": end_date.date(),
                    "source": source,
                    "current_days": current["days"],
                    "baseline_days": baseline["days"],
                    "sleep_avg_7d": round(current["sleep_avg"], 3),
                    "resting_hr_avg_7d": round(current["resting_hr_avg"], 2),
                    "steps_total_7d": round(current["steps_total"], 1),
                    "active_minutes_total_7d": round(current["active_minutes_total"], 1),
                    "sleep_avg_prev_7d": round(baseline["sleep_avg"], 3),
                    "resting_hr_avg_prev_7d": round(baseline["resting_hr_avg"], 2),
                    "steps_total_prev_7d": round(baseline["steps_total"], 1),
                    "active_minutes_total_prev_7d": round(baseline["active_minutes_total"], 1),
                    "sleep_change_pct": round(sleep_change_pct * 100, 2),
                    "hr_change_bpm": round(hr_change_bpm, 2),
                    "steps_change_pct": round(steps_change_pct * 100, 2),
                    "recovery_score": round(
                        _recovery_score(sleep_change_pct, hr_change_bpm, training_load_ratio), 3
                    ),
                    "hr_elevation_bpm": round(max(hr_change_bpm, 0.0), 2),
                    "training_load_ratio": round(training_load_ratio, 3),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    daily = pd.read_csv(INPUT)
    profiles = build_profiles(daily)
    profiles.to_csv(OUTPUT, index=False)

    print(f"Wrote {OUTPUT}")
    print(f"  Profile rows: {len(profiles):,}")
    print(f"  Users: {profiles['user_id'].nunique()}")
    print(f"  Date range: {profiles['window_end_date'].min()} to {profiles['window_end_date'].max()}")


if __name__ == "__main__":
    main()
