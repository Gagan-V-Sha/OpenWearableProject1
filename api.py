

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from explain import ExplanationEngine
from features import FEATURES
from rule_engine import INTENSIVE_THRESHOLD, REST_THRESHOLD, assess as rule_assess
from suggest import RECOVERY_BAND, SuggestionEngine

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
PROFILE_PATH = PROCESSED / "profiles_7day.csv"
DAILY_PATH = PROCESSED / "combined_daily.csv"
DEMO_PATH = PROCESSED / "user_demographics.csv"
ISO_PATH = ROOT / "models" / "isolation_forest.pkl"

app = FastAPI(title="Whyable API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

class Store:
    def __init__(self):
        self.profiles = pd.read_csv(PROFILE_PATH)
        self.daily = pd.read_csv(DAILY_PATH)
        self.demo = (
            pd.read_csv(DEMO_PATH).set_index("user_id")
            if DEMO_PATH.exists() else pd.DataFrame()
        )
        self.explainer = ExplanationEngine()
        self.suggester = SuggestionEngine()

        self.iso = None
        if ISO_PATH.exists():
            with open(ISO_PATH, "rb") as f:
                self.iso = pickle.load(f)

        self.usable = self.profiles.dropna(subset=[c for c in FEATURES if c != "rmssd_avg_7d"])

    def latest_profile(self, user_id: str) -> pd.Series:
        rows = self.usable[self.usable["user_id"] == user_id]
        if rows.empty:
            raise HTTPException(404, f"No usable profile for user '{user_id}'")
        return rows.sort_values("window_end_date").iloc[-1]

STORE = Store()

def _clean(v):

    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if pd.isna(v) else float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v

def display_name(user_id: str) -> str:
    if not user_id.startswith("LS_"):
        return user_id
    return f"Athlete {user_id[-4:].upper()}"

def score_to_band(score01: float) -> str:
    if score01 < REST_THRESHOLD:
        return "Poor"
    if score01 < INTENSIVE_THRESHOLD:
        return "Moderate"
    return "Good"

@app.get("/api/users")
def get_users():
    counts = STORE.usable.groupby("user_id").size().sort_values(ascending=False)
    users = []
    for uid in counts.index:
        demo = STORE.demo.loc[uid] if uid in STORE.demo.index else {}
        age = demo.get("age_group") if hasattr(demo, "get") else None
        users.append({
            "user_id": uid,
            "display_name": display_name(uid),
            "age_group": None if (age is None or pd.isna(age) or age == "Unknown") else str(age),
        })
    return users

def _daily_series(user_id: str, end_date: str, days: int) -> pd.DataFrame:
    d = STORE.daily[STORE.daily["user_id"] == user_id].copy()
    d = d[d["date"] <= end_date].sort_values("date").tail(days)
    return d

def _steps_change_pct(daily: pd.DataFrame) -> float | None:
    steps = daily["steps"].where(daily["steps"] > 0)
    last7, prev7 = steps.tail(7), steps.iloc[-14:-7]
    if last7.notna().sum() < 3 or prev7.notna().sum() < 3:
        return None
    prev_mean = prev7.mean()
    if not prev_mean:
        return None
    return float((last7.mean() - prev_mean) / prev_mean * 100.0)

def _load_level(ratio: float | None) -> str:
    if ratio is None:
        return "Unknown"
    if ratio > 1.2:
        return "High"
    if ratio < 0.8:
        return "Low"
    return "Balanced"

def _anomaly(row: pd.Series) -> bool | None:
    if STORE.iso is None:
        return None
    X = pd.DataFrame([{f: row.get(f, np.nan) for f in STORE.iso["features"]}])[STORE.iso["features"]]
    return bool(int(STORE.iso["model"].predict(X)[0]) == -1)

@app.get("/api/dashboard/{user_id}")
def get_dashboard(user_id: str, days: int = 30):
    row = STORE.latest_profile(user_id)
    rule = rule_assess(row)
    exp = STORE.explainer.explain(row, expertise="novice")

    fired = sorted(rule.fired(), key=lambda c: -abs(c.delta))
    basis = [f"{c.message.rstrip('.')} ({c.citation})" for c in fired[:3]]

    daily = _daily_series(user_id, row["window_end_date"], days)
    load_ratio = _clean(row.get("training_load_ratio"))

    return {
        "user_id": user_id,
        "display_name": display_name(user_id),
        "window_end_date": row["window_end_date"],
        "score": round(rule.score * 100),
        "label": score_to_band(rule.score),
        "recommendation": rule.recommendation,
        "narrative": exp.text,
        "explanation": ("Based on: " + "; ".join(basis) + ".") if basis else
                       "No strong recovery signals fired this week.",
        "ml": {
            "recommendation": exp.recommendation,
            "agrees": exp.recommendation == rule.recommendation,
            "anomaly": _anomaly(row),
        },
        "current": {f: _clean(row.get(f)) for f in FEATURES},
        "metrics": {
            "sleep": {
                "avg_hours_7d": _clean(row.get("sleep_avg_7d")),
                "change_pct": _clean(row.get("sleep_change_pct")),
            },
            "resting_hr": {
                "avg_bpm_7d": None if pd.isna(row.get("resting_hr_avg_7d"))
                              else round(float(row["resting_hr_avg_7d"])),
                "change_bpm": _clean(row.get("hr_elevation_bpm")),
            },
            "activity": {
                "steps_avg_per_day": _clean(row.get("steps_avg_7d")),
                "change_pct": _steps_change_pct(daily),
            },
            "training_load": {
                "active_minutes_7d": None if pd.isna(row.get("active_minutes_avg_7d"))
                                     else round(float(row["active_minutes_avg_7d"])),
                "level": _load_level(load_ratio),
                "change_pct": None if load_ratio is None else (load_ratio - 1.0) * 100.0,
            },
        },
        "daily": [
            {
                "date": r["date"],
                "sleep_hours": _clean(r["sleep_hours"]),
                "resting_hr": _clean(r["resting_hr"]),
                "steps": _clean(r["steps"]),
                "active_minutes": _clean(r["active_minutes"]),
            }
            for _, r in daily.iterrows()
        ],
    }

class WhatIfRequest(BaseModel):
    user_id: str
    sleep_hours: float
    training_load_ratio: float

def _counterfactual_row(row: pd.Series, sleep_hours: float, load_ratio: float) -> dict:

    cf = {f: row.get(f, np.nan) for f in FEATURES}
    cf["sleep_avg_7d"] = sleep_hours
    cf["training_load_ratio"] = load_ratio

    prev_sleep = row.get("sleep_avg_prev_7d", np.nan)
    if pd.notna(prev_sleep) and prev_sleep > 0:
        cf["sleep_change_pct"] = (sleep_hours - prev_sleep) / prev_sleep * 100.0

    prev_active = row.get("active_minutes_avg_prev_7d", np.nan)
    if pd.notna(prev_active) and prev_active > 0:
        cf["active_minutes_avg_7d"] = load_ratio * prev_active
    return cf

@app.post("/api/whatif")
def post_whatif(req: WhatIfRequest):
    row = STORE.latest_profile(req.user_id)
    base = rule_assess(row)
    cf = _counterfactual_row(row, req.sleep_hours, req.training_load_ratio)
    pred = rule_assess(cf)

    current_score = round(base.score * 100)
    predicted_score = round(pred.score * 100)
    delta = predicted_score - current_score

    sleep_delta_min = (req.sleep_hours - float(row.get("sleep_avg_7d", req.sleep_hours))) * 60.0
    parts = []
    if abs(sleep_delta_min) >= 5:
        parts.append(f"sleeping {abs(sleep_delta_min):.0f} min "
                     f"{'more' if sleep_delta_min > 0 else 'less'} per night")
    base_load = float(row.get("training_load_ratio", req.training_load_ratio))
    if abs(req.training_load_ratio - base_load) >= 0.05:
        parts.append(f"{'raising' if req.training_load_ratio > base_load else 'easing'} "
                     f"training load to {req.training_load_ratio:.2f}x")
    changes = " and ".join(parts) if parts else "keeping things as they are"

    if delta > 0:
        effect = (f"would lift your recovery to {predicted_score} "
                  f"({score_to_band(pred.score)} - {pred.recommendation}).")
    elif delta < 0:
        effect = (f"would drop your recovery to {predicted_score} "
                  f"({score_to_band(pred.score)} - {pred.recommendation}).")
    else:
        effect = f"would keep your recovery at {predicted_score} ({score_to_band(pred.score)})."

    return {
        "message": f"{changes[0].upper()}{changes[1:]} {effect}",
        "predicted_score": predicted_score,
        "predicted_label": score_to_band(pred.score),
        "current_score": current_score,
        "delta": delta,
    }

class AskRequest(BaseModel):
    user_id: str
    question: str

def _fmt_signed(v: float, digits: int = 0, suffix: str = "") -> str:
    return f"{v:+.{digits}f}{suffix}"

def _answer_recovering(row: pd.Series, rule) -> str:

    band = score_to_band(rule.score)
    lead = {
        "Poor": "Your recovery appears below your usual baseline",
        "Moderate": "Your recovery is moderate - close to your usual baseline",
        "Good": "Your recovery looks good - at or above your usual baseline",
    }[band]

    reasons = []
    sleep_chg = row.get("sleep_change_pct")
    if pd.notna(sleep_chg) and sleep_chg < -5:
        reasons.append(f"sleep duration is {abs(sleep_chg):.0f}% lower than last week")
    elif pd.notna(sleep_chg) and sleep_chg > 5:
        reasons.append(f"you are sleeping {sleep_chg:.0f}% more than last week")
    hr_chg = row.get("hr_elevation_bpm")
    if pd.notna(hr_chg) and hr_chg > 2:
        reasons.append(f"resting heart rate is elevated ({_fmt_signed(hr_chg, 1)} bpm vs baseline)")
    elif pd.notna(hr_chg) and hr_chg < -2:
        reasons.append(f"resting heart rate has dropped ({_fmt_signed(hr_chg, 1)} bpm vs baseline)")
    rmssd = row.get("rmssd_avg_7d")
    if pd.notna(rmssd):
        if rmssd < 40:
            reasons.append(f"heart-rate variability is suppressed ({rmssd:.0f} ms)")
        elif rmssd > 60:
            reasons.append(f"heart-rate variability is strong ({rmssd:.0f} ms)")

    because = (" because " + " and ".join(reasons[:2])) if reasons else ""
    return (f"{lead}{because}. Recovery score: {round(rule.score * 100)}/100 "
            f"({band}), so today's guidance is: {rule.recommendation}.")

def _answer_why(row: pd.Series, rule) -> str:
    fired = sorted(rule.fired(), key=lambda c: -abs(c.delta))
    if not fired:
        return (f"Your score of {round(rule.score * 100)} is near the neutral baseline - "
                "none of the recovery rules fired strongly this week.")
    lines = [f"- {c.message} [{'+' if c.delta > 0 else ''}{c.delta:.2f} to the score, {c.citation}]"
             for c in fired[:4]]
    return (f"Your recovery score is {round(rule.score * 100)}/100 ({score_to_band(rule.score)}), "
            f"which maps to '{rule.recommendation}'. The signals behind it:\n" + "\n".join(lines))

def _answer_bands() -> str:
    return (
        "The recovery score runs 0-100 and falls into three bands: "
        f"Poor (below {round(REST_THRESHOLD * 100)}) means your body shows clear signs of "
        "incomplete recovery, so we recommend a Rest Day; "
        f"Moderate ({round(REST_THRESHOLD * 100)}-{round(INTENSIVE_THRESHOLD * 100)}) means "
        "you are partly recovered - Light Activity is ideal; "
        f"Good ({round(INTENSIVE_THRESHOLD * 100)}+) means you are well recovered and "
        "cleared for Intensive Training."
    )

def _answer_method() -> str:
    return (
        "The score starts at 50 and transparent, research-backed rules move it up or down: "
        "heart-rate variability (Plews et al. 2013), sleep change vs your own baseline week "
        "(Fullagar et al. 2015), resting heart-rate elevation (Buchheit 2014), the acute:chronic "
        "training-load ratio (Gabbett 2016), sleep efficiency (Ohayon et al. 2017) and workout "
        "frequency (Kellmann 2010). An XGBoost model cross-checks the result, SHAP shows which "
        "features drove it, and every explanation cites only signals that actually fired - "
        "no black box."
    )

def _answer_improve(row: pd.Series) -> str:
    res = STORE.suggester.suggest(row)
    if not res.suggestions:
        return res.message
    lines = [f"{i}. {s.message}" for i, s in enumerate(res.suggestions, 1)]
    return ("Here is what the model says would actually change your outcome "
            "(counterfactuals computed with DiCE):\n" + "\n".join(lines))

def _answer_fallback(row: pd.Series, rule) -> str:
    return (
        f"Your recovery today is {round(rule.score * 100)}/100 "
        f"({score_to_band(rule.score)}) and the recommendation is {rule.recommendation}. "
        "You can ask me things like 'Am I recovering well?', 'Why is my recovery low?', "
        "'How is the score calculated?' or 'What should I change to improve it?'."
    )

@app.post("/api/ask")
def post_ask(req: AskRequest):
    row = STORE.latest_profile(req.user_id)
    rule = rule_assess(row)
    q = req.question.lower()

    def has(*words):
        return any(re.search(rf"\b{w}", q) for w in words)

    if has("improve", "change", "suggest", "better", "should i do", "what should", "advice", "fix"):
        answer = _answer_improve(row)
    elif has("calculat", "how is", "how does", "how do you", "method", "work"):
        answer = _answer_method()
    elif has("mean", "bands?", "what is a", "what does"):
        answer = _answer_bands()
    elif has("why", "reason", "explain"):
        answer = _answer_why(row, rule)
    elif has("recover", "fatigue", "tired", "how am i", "am i ok", "trend"):
        answer = _answer_recovering(row, rule)
    else:
        answer = _answer_fallback(row, rule)

    return {"answer": answer}

@app.get("/api/suggestions/{user_id}")
def get_suggestions(user_id: str):
    row = STORE.latest_profile(user_id)
    res = STORE.suggester.suggest(row)
    return {
        "user_id": user_id,
        "current_label": res.current_label,
        "current_band": res.current_band,
        "message": res.message,
        "suggestions": [
            {
                "message": s.message,
                "from_band": s.from_band,
                "to_band": s.to_band,
                "from_score": round(s.from_score * 100),
                "to_score": round(s.to_score * 100),
                "changes": [
                    {"feature": c.feature, "before": c.before, "after": c.after, "text": c.text}
                    for c in s.changes
                ],
            }
            for s in res.suggestions
        ],
    }

@app.get("/")
def health():
    return {"status": "ok", "users": int(STORE.usable["user_id"].nunique())}
