"""
Apply the explainable rule engine to 7-day profiles.

Input:
  - profiles_7day.csv
Output:
  - profiles_7day_with_rules.csv (adds recommendation + explanation)
"""

from pathlib import Path

import pandas as pd

import importlib


def main() -> None:
    root = Path(__file__).resolve().parent
    input_path = root / "profiles_7day.csv"
    output_path = root / "profiles_7day_with_rules.csv"

    df = pd.read_csv(input_path)

    # Module name starts with a digit; import via importlib.
    rule_engine = importlib.import_module("03_rule_engine")
    df_out = rule_engine.apply_rule_engine(df)

    df_out.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    print(f"  Rows: {len(df_out):,}")
    print(f"  Users: {df_out['user_id'].nunique()}")


if __name__ == "__main__":
    main()

