# HCAI Open Wearables — Explainable Recovery Assistant

Human-centred wearable recovery system that helps users understand whether they are recovering well from training. The system merges open wearable datasets, builds personal 7-day vs baseline profiles, applies transparent decision rules, and (next) adds ML predictions, LLM explanations, and a simple UI for a user study.

---

## Project goal

Many fitness apps give a single score or recommendation without showing *why*. This project builds an **explainable recovery assistant** that:

1. Uses real wearable data (sleep, heart rate, steps, activity)
2. Compares each user's **recent week to their own baseline** — not population averages
3. Produces a **transparent recommendation** (Rest / Light Activity / Intensive Training) with a human-readable explanation
4. Is designed for a later **user study** on trust, clarity, and usefulness

---

## HCAI alignment

| Requirement | How we address it |
|-------------|-----------------|
| Merge wearable data | LifeSnaps + Figshare → one daily CSV (`combined_daily.csv`) |
| Build 7/30-day profiles | Rolling **7-day windows** vs previous 7-day baseline (`profiles_7day.csv`) |
| Compare to personal baseline | Each profile row compares *current week* to *previous week* for the same user |
| Coding | Python pipeline: preprocess → profiles → rules → (ML + UI) |
| HCAI aspect | Transparent rules + text explanations; future LLM layer for natural language |
| User perspective | Recommendations framed for the user ("Am I recovering well?"); UI planned for study |

---

## Contributors

| Component | Files | By |
|-----------|-------|-----|
| **Dataset merge & preprocessing** | `preprocess.py` | **Gagan** |
| **Merged daily dataset** | `combined_daily.csv` | **Shrusti & Gagan** |
| **7-day profile builder** | `build_profiles.py`, `profiles_7day.csv` | **Shrusti** |
| **Explainable rule engine** | `03_rule_engine.py` | **Sakshi** |
| **Documentation & titles** | `README.md`, `*.TITLE.txt` | **Shrusti** |

---

## Overall pipeline

```text
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1 — Preprocessing (Gagan)                                 │
│  LifeSnaps + Figshare  →  combined_daily.csv  (1 row/user/day)  │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2 — 7-Day Profiles (Shrusti)                              │
│  Rolling windows + baseline comparison  →  profiles_7day.csv    │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3 — Rule Engine (Sakshi)                                  │
│  Transparent rules  →  recommendation + explanation             │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4 — ML Model (planned)                                    │
│  Train classifier on profile features                           │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5 — LLM Explanation Layer (planned)                       │
│  Turn rule/ML output into natural-language advice                 │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6 — Simple UI (planned)                                   │
│  Streamlit/Gradio: pick user, ask "Am I recovering well?"       │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7 — User Study (planned)                                  │
│  Evaluate trust, clarity, and usefulness of explanations          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1 — Data selection & preprocessing

**Script:** `preprocess.py` · **Author:** Gagan · **Output:** `combined_daily.csv`

### Datasets chosen

We use two open wearable datasets that both provide sleep, heart rate, and activity at daily granularity:

| Dataset | Device | Users | Duration | Prefix |
|---------|--------|-------|----------|--------|
| [LifeSnaps (Zenodo)](https://doi.org/10.5281/zenodo.6832242) | Fitbit Sense | 71 | ~4 months | `LS_` |
| [Figshare HRV + Sleep](https://doi.org/10.6084/m9.figshare.28509740) | Samsung Galaxy Active 2 | 49 | ~4 weeks | `FS_` |

**Total:** 120 users, 8,782 daily rows.

Other datasets (BIDSleep, raw_data.zip) were considered but not used — LifeSnaps and Figshare were sufficient and easier to align.

### What preprocessing does

1. **LifeSnaps** — reads `daily_fitbit_sema_df_unprocessed.csv`:
   - `sleep_hours` = `minutesAsleep / 60`
   - `active_minutes` = lightly + moderately + very active minutes
   - Keeps `resting_hr`, `steps`, `sleep_efficiency`, `rmssd`
   - Prefixes user IDs with `LS_`

2. **Figshare** — merges two files:
   - `sleep_diary.csv` → sleep duration and efficiency per day
   - `sensor_hrv_filtered.csv` → aggregated to daily: mean HR, sum steps, mean rmssd
   - Prefixes user IDs with `FS_`

3. **Combine** — concatenates both sources into one schema, sorted by `user_id` + `date`.

### Output schema (`combined_daily.csv`)

| Column | Description |
|--------|-------------|
| `user_id` | Unique user (`LS_*` or `FS_*`) |
| `date` | Calendar date |
| `sleep_hours` | Total sleep (hours) |
| `resting_hr` | Resting / mean heart rate (bpm) |
| `steps` | Daily step count |
| `sleep_efficiency` | Sleep efficiency (%) |
| `active_minutes` | Active time (minutes) |
| `rmssd` | HRV metric (ms) |
| `source` | `lifesnaps` or `figshare` |

---

## Step 2 — 7-day profile builder

**Script:** `build_profiles.py` · **Author:** Shrusti · **Output:** `profiles_7day.csv`

This is the core feature-engineering step. Instead of looking at a single day, we build **rolling weekly profiles** and compare each user to **their own recent baseline**.

### Window logic

For every user and every date they have data:

```
Timeline:  |---- baseline week (7 days) ----|---- current week (7 days) ----|
           baseline_start              baseline_end (= current_start)   end_date
```

- **Current window:** the 7 days ending on `window_end_date` (exclusive start, inclusive end)
- **Baseline window:** the 7 days immediately before the current window
- **Minimum data:** at least **4 days** of readings required in each window (handles missing wearable days)

This gives a **personal baseline comparison** — e.g. "your sleep this week vs your sleep last week", not vs other users.

### Aggregated metrics per window

| Metric | Current window column | Baseline window column |
|--------|----------------------|------------------------|
| Average sleep (hours) | `sleep_avg_7d` | `sleep_avg_prev_7d` |
| Average resting HR (bpm) | `resting_hr_avg_7d` | `resting_hr_avg_prev_7d` |
| Total steps | `steps_total_7d` | `steps_total_prev_7d` |
| Total active minutes | `active_minutes_total_7d` | `active_minutes_total_prev_7d` |

### Change / derived features

| Feature | Formula | Meaning |
|---------|---------|---------|
| `sleep_change_pct` | `(current − baseline) / baseline × 100` | Sleep trend vs own baseline |
| `hr_change_bpm` | `current HR − baseline HR` | Heart rate elevation (bpm) |
| `steps_change_pct` | `(current − baseline) / baseline × 100` | Activity trend |
| `training_load_ratio` | `current active min / baseline active min` | Relative training load |
| `hr_elevation_bpm` | `max(hr_change_bpm, 0)` | Only positive HR elevation |
| `recovery_score` | Weighted score (see below) | 0–1 readiness indicator |

### Recovery score formula

Starting score: **0.55**, then adjusted:

| Signal | Condition | Adjustment |
|--------|-----------|------------|
| Sleep drop | `sleep_change_pct < −5%` | −0.20 |
| Sleep gain | `sleep_change_pct > +5%` | +0.10 |
| HR elevated | `hr_change_bpm > +3 bpm` | −0.20 |
| HR lower | `hr_change_bpm < −2 bpm` | +0.10 |
| High load | `training_load_ratio > 1.30` | −0.15 |
| Low load | `training_load_ratio < 0.85` | +0.05 |

Final score clipped to **[0.0, 1.0]**.

### Output stats

- **7,533 profile rows** (one per user per valid week-end date)
- **120 users**
- Key columns used downstream: `recovery_score`, `hr_elevation_bpm`, `training_load_ratio`

---

## Step 3 — Explainable rule engine

**Script:** `03_rule_engine.py` · **Author:** Sakshi

Takes profile features and returns a **recommendation** plus a **text explanation** — no black box.

### Recommendation rules

| Recommendation | Conditions (any triggers Rest; all must hold for Intensive) |
|----------------|-------------------------------------------------------------|
| **Rest Day** | `recovery_score < 0.40` **OR** `hr_elevation_bpm > 5` **OR** `training_load_ratio > 1.5` |
| **Intensive Training** | `recovery_score ≥ 0.58` **AND** `hr_elevation_bpm < 3` **AND** `training_load_ratio < 1.2` |
| **Light Activity** | Everything else (moderate zone) |

### Explanation rules

The engine collects human-readable reasons:

| Signal | Reason added |
|--------|-------------|
| `recovery_score < 0.45` | "low recovery score" |
| `hr_elevation_bpm > 3` | "elevated heart rate" |
| `training_load_ratio > 1.2` | "high training load" |
| `0.5 ≤ training_load_ratio < 1.0` | "balanced training load" |
| `recovery_score > 0.60` | "strong recovery readiness" |

If no specific reasons match, a default explanation is used per recommendation level.

### Example output columns

| Column | Example |
|--------|---------|
| `recommendation` | `Light Activity` |
| `explanation` | `Recommendation based on: elevated heart rate, high training load` |

---

## Steps 4–7 — Planned next

| Step | Goal | Status |
|------|------|--------|
| **4 — ML model** (`train_model.py`) | Train a classifier on `profiles_7day.csv` features; compare with rule engine | Not started |
| **5 — LLM layer** | Generate natural-language explanations from rule/ML output + user context | Not started |
| **6 — UI** | Simple Streamlit/Gradio app: select user, view profile, get recommendation | Not started |
| **7 — User study** | Evaluate whether explanations are understandable and trustworthy | End of project |

---

## Repository files

| # | File | Title | Description | By |
|---|------|-------|-------------|-----|
| 01 | `preprocess.py` | **Data Preprocessing & Dataset Merge** | Combines LifeSnaps + Figshare into one daily CSV | Gagan |
| 01 | `combined_daily.csv` | **Merged Daily Wearable Dataset** | 120 users, 8,782 rows — sleep, HR, steps per day | Shrusti & Gagan |
| 02 | `build_profiles.py` | **7-Day Profile Builder** | Last 7 days vs previous 7 days baseline comparison | Shrusti |
| 02 | `profiles_7day.csv` | **7-Day Recovery Profiles** | 7,533 weekly profiles with recovery features | Shrusti |
| 03 | `03_rule_engine.py` | **Explainable Rule Engine** | Transparent rules → recommendation + explanation | Sakshi |

---

## Run order

```powershell
# Step 1 — only needed when re-building from raw source CSVs
python preprocess.py

# Step 2 — builds profiles from combined_daily.csv
python build_profiles.py

# Step 3 — apply rules (example usage in Python)
python -c "import pandas as pd; from importlib import import_module; m=import_module('03_rule_engine'); df=pd.read_csv('profiles_7day.csv'); print(m.apply_rule_engine(df).head())"
```

Place raw source files under `data/lifesnaps/` and `data/figshare/` before running `preprocess.py`. The committed `combined_daily.csv` and `profiles_7day.csv` can be used directly without re-running.

---

## Data sources

- [LifeSnaps (Zenodo)](https://doi.org/10.5281/zenodo.6832242) — Fitbit Sense, 71 users, ~4 months
- [Figshare HRV + Sleep](https://doi.org/10.6084/m9.figshare.28509740) — Samsung Galaxy Active 2, 49 users, ~4 weeks

---

## Known limitations

- **Figshare resting HR** is the mean HR from 5-minute sensor segments, not a true resting HR — values can run higher (~80–90 bpm) than LifeSnaps resting HR.
- **Missing days:** windows need ≥4 days of data; sparse users produce fewer profile rows.
- **Cross-device comparison:** LifeSnaps and Figshare use different devices; profiles are always compared within the same user, not across users.
- **30-day profiles:** not yet implemented; current pipeline uses 7-day windows only.

---

## Team

**HCAI Group — Open Wearables Project**
