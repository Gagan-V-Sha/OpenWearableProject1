def build_prompt(row):

    return f"""
You are an AI recovery coach.

Wearable Metrics:

Recovery Score: {row['recovery_score']}

Heart Rate Elevation:
{row['hr_elevation_bpm']} bpm

Training Load Ratio:
{row['training_load_ratio']}

7-Day Sleep Average:
{row['sleep_avg_7d']} hours

7-Day Activity Minutes:
{row['active_minutes_total_7d']}

Rule Engine Recommendation:
{row['recommendation']}

Reason:
{row['explanation']}

Explain:

1. What the metrics mean
2. Why this recommendation was assigned
3. Practical recovery advice

Keep response under 120 words.

Do not diagnose medical conditions.
"""
