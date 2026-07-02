Data files
==========

data/raw/daily_fitbit_sema_df_unprocessed.csv
  Original LifeSnaps daily Fitbit + SEMA export (7,410 rows, 71 users).

data/processed/daily_cleaned.csv
  Cleaned version produced by src/data/loader.py:
  - column names harmonised
  - sleep durations converted to hours (sleep_duration_h, minutes_asleep_h)
  - duplicate user-days merged
  - impossible values set to missing (e.g. resting HR, SpO2, steps)
  This is the table the rule engine and ML model read at runtime.

data/raw/personality.csv, data/raw/stai.csv
  Optional survey files from the same LifeSnaps study (not required to run the app).

To regenerate the cleaned file from raw:
  python -c "from src.data.loader import load_daily; load_daily().to_csv('data/processed/daily_cleaned.csv', index=False)"
