import sys
from pathlib import Path
from dataclasses import asdict
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline import respond, _load_iso
from explain import ExplanationEngine
from features import FEATURES

app = FastAPI(
    title="OpenWearable HCAI API",
    description="Human-Centered AI Recovery Recommendation & Explanation API",
    version="0.1.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROFILE_PATH = PROCESSED_DIR / "profiles_7day.csv"
REPORT_PATH = PROCESSED_DIR / "fairness_report.json"

class ProfileInput(BaseModel):
    sleep_change_pct: Optional[float] = None
    hr_elevation_bpm: Optional[float] = None
    training_load_ratio: Optional[float] = None
    sleep_avg_7d: Optional[float] = None
    resting_hr_avg_7d: Optional[float] = None
    steps_avg_7d: Optional[float] = None
    active_minutes_avg_7d: Optional[float] = None
    rmssd_avg_7d: Optional[float] = None
    sleep_efficiency_avg_7d: Optional[float] = None
    workouts_count: Optional[float] = None
    rmssd_missing: Optional[float] = 0.0

@app.get("/")
def read_root():
    return {
        "status": "ok",
        "service": "OpenWearable HCAI API",
        "endpoints": [
            "/api/profiles",
            "/api/profile/{user_id}",
            "/api/respond",
            "/api/audit"
        ]
    }

@app.get("/api/profiles")
def get_profiles(limit: int = Query(50, ge=1, le=500)):
    if not PROFILE_PATH.exists():
        raise HTTPException(status_code=404, detail="Profiles dataset not found.")
    df = pd.read_csv(PROFILE_PATH)
    # Replace NaN with None for valid JSON serialization
    df_clean = df.replace({np.nan: None})
    return {
        "count": len(df_clean),
        "profiles": df_clean.head(limit).to_dict(orient="records")
    }

@app.get("/api/profile/{user_id}")
def get_profile_explanation(
    user_id: int,
    expertise: str = Query("expert", regex="^(novice|expert)$"),
    use_llm: bool = Query(False)
):
    if not PROFILE_PATH.exists():
        raise HTTPException(status_code=404, detail="Profiles dataset not found.")
    df = pd.read_csv(PROFILE_PATH)
    user_rows = df[df["user_id"] == user_id]
    if user_rows.empty:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
    
    row = user_rows.iloc[-1]
    try:
        r = respond(row, expertise=expertise, use_llm=use_llm)
        exp_dict = asdict(r["explanation"])
        return {
            "user_id": int(user_id),
            "window_end_date": str(row.get("window_end_date", "")),
            "anomaly": r["anomaly"],
            "explanation": exp_dict,
            "metrics": {f: (None if pd.isna(row.get(f)) else float(row.get(f))) for f in FEATURES}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/respond")
def analyze_custom_profile(
    data: ProfileInput,
    expertise: str = Query("expert", regex="^(novice|expert)$"),
    use_llm: bool = Query(False)
):
    row_dict = data.dict()
    row_series = pd.Series(row_dict)
    try:
        r = respond(row_series, expertise=expertise, use_llm=use_llm)
        exp_dict = asdict(r["explanation"])
        return {
            "anomaly": r["anomaly"],
            "explanation": exp_dict,
            "metrics": row_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/audit")
def get_fairness_audit():
    if not REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="Fairness report not found.")
    import json
    with open(REPORT_PATH, "r") as f:
        return json.load(f)

# Expose handler for Vercel serverless
handler = app
