import pandas as pd
import google.generativeai as genai

genai.configure(api_key="API_KEY")

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

df = pd.read_csv(
    "profiles_7day_with_rules.csv"
)

reports = []

for _, row in df.iterrows():

    prompt = f"""
    You are an AI recovery coach.

    Recovery Score:
    {row['recovery_score']}

    Heart Rate Elevation:
    {row['hr_elevation_bpm']}

    Training Load Ratio:
    {row['training_load_ratio']}

    Recommendation:
    {row['recommendation']}

    Explanation:
    {row['explanation']}

    Generate a personalized recovery report.
    """

    response = model.generate_content(
        prompt
    )

    reports.append(
        response.text
    )

df["llm_report"] = reports

df.to_csv(
    "profiles_7day_with_llm.csv",
    index=False
)
