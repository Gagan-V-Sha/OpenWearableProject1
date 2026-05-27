# HCAI Open Wearables — Preprocessed Dataset

Daily-level wearable data merged from two public research datasets for the HCAI recovery/fatigue explainability project.

## Sources

| Source | Device | Users | File |
|--------|--------|-------|------|
| [LifeSnaps](https://doi.org/10.5281/zenodo.6832242) | Fitbit Sense | 71 | `data/lifesnaps/.../daily_fitbit_sema_df_unprocessed.csv` |
| [Figshare HRV + Sleep](https://doi.org/10.6084/m9.figshare.28509740) | Samsung Galaxy Active 2 | 49 | `data/figshare/sleep_diary.csv`, `sensor_hrv_filtered.csv` |

## Output

`data/processed/combined_daily.csv` — one row per user per day:

| Column | Description |
|--------|-------------|
| `user_id` | Prefixed ID (`LS_*` LifeSnaps, `FS_*` Figshare) |
| `date` | Calendar date |
| `sleep_hours` | Sleep duration (hours) |
| `resting_hr` | Resting / mean heart rate (bpm) |
| `steps` | Daily step count |
| `sleep_efficiency` | Sleep efficiency (%) |
| `active_minutes` | Active time (minutes) |
| `rmssd` | HRV RMSSD where available |
| `source` | `lifesnaps` or `figshare` |

## Reproduce

```bash
pip install -r requirements.txt
python preprocess.py
```

Raw source CSVs are not included in this repo (download from Zenodo/Figshare). Only the merged output and preprocessing script are versioned.
