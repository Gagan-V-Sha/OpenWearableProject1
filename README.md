# HCAI Open Wearables — Explainable Recovery Assistant

Human-centred wearable recovery system using LifeSnaps (Fitbit) + Figshare (Samsung) data.

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

## Repository files

| # | File | Title | Description | By |
|---|------|-------|-------------|-----|
| 01 | `preprocess.py` | **Data Preprocessing & Dataset Merge** | Combines LifeSnaps + Figshare into one daily CSV | Gagan |
| 01 | `combined_daily.csv` | **Merged Daily Wearable Dataset** | 120 users, 8,782 rows — sleep, HR, steps per day | Shrusti & Gagan |
| 02 | `build_profiles.py` | **7-Day Profile Builder** | Last 7 days vs previous 7 days baseline comparison | Shrusti |
| 02 | `profiles_7day.csv` | **7-Day Recovery Profiles** | 7,533 weekly profiles with recovery features | Shrusti |
| 03 | `03_rule_engine.py` | **Explainable Rule Engine** | Transparent rules → recommendation + explanation | Sakshi |

---

## Pipeline

```text
preprocess.py  →  combined_daily.csv
       ↓
build_profiles.py  →  profiles_7day.csv
       ↓
03_rule_engine.py  →  recommendation + explanation
       ↓
(train_model.py + UI — coming next)
```

---

## Run order

```powershell
python preprocess.py      # only if re-building from raw source CSVs
python build_profiles.py  # builds profiles_7day.csv from combined_daily.csv
```

---

## Data sources

- [LifeSnaps (Zenodo)](https://doi.org/10.5281/zenodo.6832242) — Fitbit Sense, 71 users
- [Figshare HRV + Sleep](https://doi.org/10.6084/m9.figshare.28509740) — Samsung watch, 49 users

---

## Team

**HCAI Group — Open Wearables Project**

- **Shrusti** — dataset (with Gagan), 7-day profiles, documentation
- **Gagan** — preprocessing script, dataset (with Shrusti)
- **Sakshi** — explainable rule engine
