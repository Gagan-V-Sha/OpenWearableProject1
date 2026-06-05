from pathlib import Path

import pandas as pd


def recovery_recommendation(row):

    reasons = []

    recovery = row['recovery_score']
    hr = row['hr_elevation_bpm']
    load = row['training_load_ratio']
    sleep = row['sleep_avg_7d']

    # Handle missing values
    if pd.isna(load):
        load = 1.0

    if pd.isna(hr):
        hr = 0

    if pd.isna(sleep):
    sleep = 7.0

    # Recommendation logic
   if (
    recovery < 0.40
    or hr > 5
    or load > 1.5
    or sleep < 6
):
    recommendation = "Rest Day"

elif (
    recovery >= 0.58
    and hr < 3
    and load < 1.2
    and sleep >= 7
):
    recommendation = "Intensive Training"

else:
    recommendation = "Light Activity"

    # Explanation logic
    if recovery < 0.45:
        reasons.append("low recovery score")

    if hr > 3:
        reasons.append("elevated heart rate")

    if load > 1.2:
        reasons.append("high training load")

    if 0.5 <= load < 1:
        reasons.append("balanced training load")

    if recovery > 0.60:
        reasons.append("strong recovery readiness")

    if sleep >= 7.5:
    reasons.append("adequate sleep")

    # Human explanation
    if len(reasons) == 0:

        if recommendation == "Intensive Training":
            explanation = (
                "Strong recovery readiness detected "
                "with stable physiological indicators."
            )

        elif recommendation == "Light Activity":
            explanation = (
                "Moderate recovery readiness detected. "
                "Light activity is recommended."
            )

        else:
            explanation = (
                "Recovery signals suggest additional "
                "rest may be beneficial."
            )

    else:
        explanation = (
            "Recommendation based on: "
            + ", ".join(reasons)
        )

    return pd.Series([
        recommendation,
        explanation
    ])


def apply_rule_engine(df):

    df[
        ['recommendation',
         'explanation']
    ] = df.apply(
        recovery_recommendation,
        axis=1
    )

    return df


def main() -> None:
    """
    CLI entrypoint:
      python 03_rule_engine.py

    Reads profiles_7day.csv and writes profiles_7day_with_rules.csv.
    """
    root = Path(__file__).resolve().parent
    input_path = root / "profiles_7day.csv"
    output_path = root / "profiles_7day_with_rules.csv"

    df = pd.read_csv(input_path)
    df_out = apply_rule_engine(df)
    df_out.to_csv(output_path, index=False)

    print(f"Wrote {output_path}")
    print(f"  Rows: {len(df_out):,}")
    print(f"  Users: {df_out['user_id'].nunique()}")


if __name__ == "__main__":
    main()
