

from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from features import FEATURES
from rule_engine import INTENSIVE_THRESHOLD, REST_THRESHOLD, assess as rule_assess

try:
    from explain import ExplanationEngine
except ImportError:
    ExplanationEngine = None

try:
    from suggest import RECOVERY_BAND, SuggestionEngine
except ImportError:
    RECOVERY_BAND = {
        "Rest Day": "Poor",
        "Light Activity": "Moderate",
        "Intensive Training": "Good",
    }
    SuggestionEngine = None

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
        self.explainer = None
        if ExplanationEngine is not None:
            try:
                self.explainer = ExplanationEngine()
            except Exception:
                self.explainer = None
        self.suggester = None
        if SuggestionEngine is not None:
            try:
                self.suggester = SuggestionEngine()
            except Exception:
                self.suggester = None

        self.iso = None
        if ISO_PATH.exists():
            try:
                with open(ISO_PATH, "rb") as f:
                    self.iso = pickle.load(f)
            except Exception:
                self.iso = None

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
        gender = demo.get("gender") if hasattr(demo, "get") else None
        users.append({
            "user_id": uid,
            "display_name": display_name(uid),
            "age_group": None if (age is None or pd.isna(age) or age == "Unknown") else str(age),
            "gender": None if (gender is None or pd.isna(gender) or gender == "Unknown") else str(gender),
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

def _rule_narrative(rule) -> str:
    fired = sorted(rule.fired(), key=lambda c: -abs(c.delta))
    rec = rule.recommendation
    lead = {
        "Rest Day": "Your body looks like it needs a rest today.",
        "Light Activity": "A light, easy session is the sweet spot for you today.",
        "Intensive Training": "You look well recovered and ready for a hard session.",
    }[rec]
    reasons = [c.message.rstrip(".") for c in fired[:2]]
    why = (" Mainly because " + "; and ".join(r.lower() for r in reasons) + ".") if reasons else ""
    return lead + why

@app.get("/api/dashboard/{user_id}")
def get_dashboard(user_id: str, days: int = 30):
    row = STORE.latest_profile(user_id)
    rule = rule_assess(row)
    if STORE.explainer is not None:
        exp = STORE.explainer.explain(row, expertise="novice")
        narrative = exp.text
        ml_rec = exp.recommendation
    else:
        narrative = _rule_narrative(rule)
        ml_rec = rule.recommendation

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
        "narrative": narrative,
        "explanation": ("Based on: " + "; ".join(basis) + ".") if basis else
                       "No strong recovery signals fired this week.",
        "ml": {
            "recommendation": ml_rec,
            "agrees": ml_rec == rule.recommendation,
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

class AskTurn(BaseModel):
    role: str
    text: str

class AskRequest(BaseModel):
    user_id: str
    question: str
    history: list[AskTurn] = []

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

def _why_low_recovery(row: pd.Series, rule) -> str:
    score = round(rule.score * 100)
    band = score_to_band(rule.score)
    neg = sorted([c for c in rule.fired() if c.delta < 0], key=lambda c: c.delta)
    pos = sorted([c for c in rule.fired() if c.delta > 0], key=lambda c: -c.delta)
    parts = [f"Your recovery is {score}/100 ({band})"]
    if band == "Poor":
        parts[0] += " — well below the 40-point threshold"
    parts[0] += "."
    if neg:
        drivers = ", ".join(c.message.rstrip(".").lower() for c in neg[:3])
        parts.append(f"The main factors pulling it down: {drivers}.")
    if pos:
        parts.append(f"On the plus side: {pos[0].message.rstrip('.').lower()}.")
    return " ".join(parts)

def _why_rest_day(row: pd.Series, rule) -> str:
    score = round(rule.score * 100)
    band = score_to_band(rule.score)
    if rule.recommendation != "Rest Day":
        return (f"Today's guidance is actually {rule.recommendation}, not a full rest day — "
                f"your score is {score}/100 ({band}).")
    neg = sorted([c for c in rule.fired() if c.delta < 0], key=lambda c: c.delta)
    drivers = ", ".join(c.message.rstrip(".").lower() for c in neg[:2]) if neg else (
        "your recovery signals are below baseline"
    )
    return (f"A rest day is recommended because your score is {score}/100 ({band} — below 40). "
            f"The main drivers: {drivers}. Light movement is fine, but hard training would add strain.")

def _answer_poor_mean(rule) -> str:
    score = round(rule.score * 100)
    return (
        f"A Poor score (below {round(REST_THRESHOLD * 100)}) means incomplete recovery — "
        f"we recommend a Rest Day. Yours is {score}/100 right now. "
        f"Moderate ({round(REST_THRESHOLD * 100)}–{round(INTENSIVE_THRESHOLD * 100)}) = light activity; "
        f"Good ({round(INTENSIVE_THRESHOLD * 100)}+) = ready for hard training."
    )

def _answer_why(row: pd.Series, rule) -> str:
    return _why_low_recovery(row, rule)

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
        f"I'm not sure how to answer that. Your recovery today is {round(rule.score * 100)}/100 "
        f"({score_to_band(rule.score)}) — guidance: {rule.recommendation}. "
        "Try a suggestion below, or ask 'why is my recovery low?'"
    )

def _open_chat_answer(row: pd.Series, rule, q: str) -> str | None:
    if _has_word(q, "who are you", "what are you"):
        return (
            "I'm Whyable — I explain your recovery scores from wearable data. "
            "I don't diagnose or give medical advice; I translate your metrics into plain guidance."
        )
    if re.search(r"\bwho am i\b", q):
        return (
            "I don't know your identity — only this week's wearable metrics for the selected athlete. "
            f"Right now that profile shows {round(rule.score * 100)}/100 ({score_to_band(rule.score)})."
        )
    if _has_word(q, "eat", "food", "meal", "diet", "nutrition", "snack"):
        return (
            "I can't recommend specific foods. I focus on recovery signals — sleep, HRV, and training load. "
            "Ask 'what should I change?' for training and sleep adjustments backed by your data."
        )
    if re.search(r"\bwhere am i\b", q) or _has_word(q, "location", "where i am"):
        return (
            "I don't have location data — only this athlete's wearable recovery metrics for the past week."
        )
    if re.search(r"rec[eov]+ry", q) and _has_word(q, "good", "gud", "great", "fine", "well", "bad", "poor"):
        score = round(rule.score * 100)
        band = score_to_band(rule.score)
        if _has_word(q, "good", "gud", "great", "fine", "well") and band != "Good":
            return (
                f"Your wearable data shows {score}/100 ({band}) today — we'd still recommend "
                f"{rule.recommendation}. Ask 'why is my recovery low?' to see what's driving that."
            )
        if _has_word(q, "bad", "poor") and band == "Good":
            return (
                f"Actually your data looks strong — {score}/100 ({band}). "
                f"Today's guidance is {rule.recommendation}."
            )
        return _answer_recovering(row, rule)
    return None

def _ask_facts(row: pd.Series, rule) -> dict:
    facts = {
        "recovery_score": round(rule.score * 100),
        "band": score_to_band(rule.score),
        "recommendation": rule.recommendation,
        "rules_fired": [
            {"signal": c.signal, "message": c.message, "delta": round(float(c.delta), 3)}
            for c in rule.fired()
        ],
        "metrics": {
            key: None if pd.isna(row.get(key)) else round(float(row[key]), 2)
            for key in (
                "sleep_change_pct", "hr_elevation_bpm", "training_load_ratio",
                "sleep_avg_7d", "resting_hr_avg_7d", "rmssd_avg_7d",
                "sleep_efficiency_avg_7d", "workouts_count", "steps_avg_7d",
            )
        },
    }
    if STORE.explainer is not None:
        try:
            ev = STORE.explainer.build_evidence(row)
            facts["ml_recommendation"] = ev["recommendation"]
            facts["ml_confidence"] = round(float(ev["ml_confidence"]), 3)
            facts["top_shap"] = [
                {
                    "feature": c.feature,
                    "value": None if pd.isna(c.value) else round(float(c.value), 2),
                    "shap": round(float(c.shap_value), 3),
                }
                for c in ev["shap_ranking"]
            ]
        except Exception:
            pass
    return facts

def _try_ask_llm(row: pd.Series, rule, question: str,
                 history: list[AskTurn] | None = None) -> str | None:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        facts = _ask_facts(row, rule)
        history = history or []
        history_block = ""
        if history:
            lines = [f"{t.role.title()}: {t.text}" for t in history[-8:]]
            history_block = "RECENT_CHAT:\n" + "\n".join(lines) + "\n\n"
        prompt = (
            "You are Whyable, a recovery coach in the OpenWearable app. "
            "Answer the user's latest question directly in plain language (no jargon, no diagnoses). "
            "Use ONLY the facts in USER_DATA — do not invent numbers or metrics. "
            "Rule deltas in USER_DATA are on a 0–1 scale (not the 0–100 score); "
            "do not quote them as 'points' on the 0–100 score. "
            "Do NOT greet the user (no 'Hi', 'Hello', 'Hey'). "
            "Do NOT introduce yourself unless they explicitly ask who or what you are. "
            "If RECENT_CHAT shows you already introduced yourself, never repeat that. "
            "For follow-up questions, answer only what was asked — stay concise. "
            "If the question is off-topic (food, identity, etc.), give a brief redirect "
            "without re-introducing yourself. "
            "2-3 short sentences unless they ask for detail.\n\n"
            f"{history_block}"
            f"USER_QUESTION: {question}\n\n"
            f"USER_DATA:\n{json.dumps(facts, indent=2)}"
        )
        resp = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=prompt,
        )
        text = (resp.text or "").strip()
        return text or None
    except Exception as e:
        print(f"[ask] Gemini unavailable ({e}); using template.")
        return None

def _answer_pushback() -> str:
    return (
        "Sorry that wasn't clear. Try 'why is my recovery low?', 'why a rest day?', "
        "or 'what should I change to improve it?'"
    )

def _answer_affirmative(row: pd.Series, rule) -> str:
    score = round(rule.score * 100)
    band = score_to_band(rule.score)
    return (
        f"Your score is {score}/100 ({band}) — today's guidance is {rule.recommendation}. "
        "Ask 'why is my recovery low?' or 'what should I change?' for more detail."
    )

def _has_word(q: str, *words: str) -> bool:
    return any(re.search(rf"\b{w}", q) for w in words)

def _template_answer(row: pd.Series, rule, q: str) -> str | None:
    if _has_word(q, "improve", "change", "suggest", "better", "should i do", "what should i",
                 "advice", "fix") and not _has_word(q, "eat", "food", "meal", "diet", "nutrition"):
        return _answer_improve(row)
    if _has_word(q, "calculat", "how is the score", "how does the score", "how do you",
                 "how it work", "scoring method", "score work"):
        return _answer_method()
    if _has_word(q, "poor") and _has_word(q, "mean", "what does", "what is"):
        return _answer_poor_mean(rule)
    if _has_word(q, "mean", "bands?", "what is a", "what does"):
        return _answer_bands()
    if _has_word(q, "rest day", "rest days") and _has_word(q, "why", "reason"):
        return _why_rest_day(row, rule)
    if _has_word(q, "why", "reason", "explain") and _has_word(
            q, "low", "recover", "recovery", "today", "score", "poor"):
        return _why_low_recovery(row, rule)
    if _has_word(q, "why", "reason", "explain"):
        return _answer_why(row, rule)
    if _has_word(q, "recover", "fatigue", "tired", "trend") or re.search(
            r"\bhow am i\b", q) or re.search(r"\bam i ok\b", q):
        return _answer_recovering(row, rule)
    return None

def _is_pushback(q: str) -> bool:
    return bool(re.match(
        r"^(no+|noo+|nah|nope|wrong|incorrect|not right|that's wrong|thats wrong|"
        r"i don'?t understand|didn'?t help)\W*$",
        q.strip(),
    ))

def _is_affirmative(q: str) -> bool:
    return bool(re.match(
        r"^(yes+|yeah|yep|yup|y|ok|okay|sure|thanks|thank you|got it|cool)\W*$",
        q.strip(),
    ))

@app.post("/api/ask")
def post_ask(req: AskRequest):
    row = STORE.latest_profile(req.user_id)
    rule = rule_assess(row)
    q = req.question.lower().strip()

    if _is_pushback(q):
        return {"answer": _answer_pushback(), "source": "template"}

    if _is_affirmative(q):
        return {"answer": _answer_affirmative(row, rule), "source": "template"}

    template = _template_answer(row, rule, q)
    if template is not None:
        return {"answer": template, "source": "template"}

    open_answer = _open_chat_answer(row, rule, q)
    if open_answer is not None:
        return {"answer": open_answer, "source": "template"}

    llm_answer = _try_ask_llm(row, rule, req.question, req.history)
    if llm_answer:
        return {"answer": llm_answer, "source": "llm"}

    return {"answer": _answer_fallback(row, rule), "source": "fallback"}

@app.get("/api/suggestions/{user_id}")
def get_suggestions(user_id: str):
    if STORE.suggester is None:
        raise HTTPException(503, "Suggestions are not available on this deployment.")
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

FEATURE_LABELS = {
    "sleep_change_pct": "Sleep change vs baseline",
    "hr_elevation_bpm": "Resting HR elevation",
    "training_load_ratio": "Training load (ACWR)",
    "sleep_avg_7d": "Average sleep",
    "resting_hr_avg_7d": "Average resting HR",
    "steps_avg_7d": "Average daily steps",
    "active_minutes_avg_7d": "Average active minutes",
    "rmssd_avg_7d": "HRV (RMSSD)",
    "sleep_efficiency_avg_7d": "Sleep efficiency",
    "workouts_count": "Workouts this week",
    "rmssd_missing": "HRV data availability",
}

def _fmt_feature(feature: str, value) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    v = float(value)
    units = {
        "sleep_change_pct": f"{v:+.0f}%",
        "hr_elevation_bpm": f"{v:+.1f} bpm",
        "training_load_ratio": f"{v:.2f}x",
        "sleep_avg_7d": f"{v:.1f} h",
        "resting_hr_avg_7d": f"{v:.0f} bpm",
        "rmssd_avg_7d": f"{v:.0f} ms",
        "sleep_efficiency_avg_7d": f"{v:.0f}%",
        "workouts_count": f"{v:.0f}",
    }
    return units.get(feature, f"{v:.1f}")

@app.get("/api/monitoring/{user_id}")
def get_monitoring(user_id: str, days: int = 120):
    rows = STORE.usable[STORE.usable["user_id"] == user_id].sort_values("window_end_date")
    if rows.empty:
        raise HTTPException(404, f"No usable profile for user '{user_id}'")
    rows = rows.tail(days)
    return {
        "user_id": user_id,
        "score_history": [
            {
                "date": r["window_end_date"],
                "score": round(float(r["rule_score"]) * 100),
                "recommendation": r["rule_recommendation"],
            }
            for _, r in rows.iterrows()
        ],
    }

@app.get("/api/audit/{user_id}")
def get_audit(user_id: str):
    row = STORE.latest_profile(user_id)
    rule = rule_assess(row)
    score = round(rule.score * 100)
    fired = sorted(rule.fired(), key=lambda c: -abs(c.delta))

    demo = STORE.demo.loc[user_id] if user_id in STORE.demo.index else None

    def _demo(key):
        if demo is None:
            return None
        v = demo.get(key)
        return None if (v is None or pd.isna(v) or v == "Unknown") else str(v)

    checks = [
        {
            "signal": FEATURE_LABELS.get(c.signal, c.signal),
            "value": _fmt_feature(c.signal, c.value),
            "message": c.message,
            "delta": round(float(c.delta), 3),
            "citation": c.citation,
            "passed": c.delta > 0,
        }
        for c in fired
    ]

    # "What moved the score": SHAP attributions when the ML stack is up,
    # otherwise the transparent rule deltas.
    moved = {
        "source": "rules",
        "items": [
            {
                "label": FEATURE_LABELS.get(c.signal, c.signal),
                "value": _fmt_feature(c.signal, c.value),
                "impact": round(float(c.delta), 3),
            }
            for c in fired
        ],
    }
    explanation = {"text": _rule_narrative(rule), "source": "rules", "faithfulness_score": None}
    ml_confidence = None
    if STORE.explainer is not None:
        try:
            ev = STORE.explainer.build_evidence(row)
            ml_confidence = round(float(ev["ml_confidence"]), 2)
            moved = {
                "source": "shap",
                "items": [
                    {
                        "label": FEATURE_LABELS.get(c.feature, c.feature),
                        "value": _fmt_feature(c.feature, c.value),
                        "impact": round(float(c.shap_value), 3),
                    }
                    for c in ev["shap_ranking"]
                ],
            }
            use_llm = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
            exp = STORE.explainer.explain(row, expertise="novice", use_llm=use_llm)
            explanation = {
                "text": exp.text,
                "source": exp.source,
                "faithfulness_score": None if exp.faithfulness_score is None
                                      else round(exp.faithfulness_score, 2),
            }
        except Exception:
            pass

    fairness = None
    fairness_path = PROCESSED / "fairness_report.json"
    if fairness_path.exists():
        try:
            fairness = json.loads(fairness_path.read_text())
        except Exception:
            fairness = None

    history = (
        STORE.usable[STORE.usable["user_id"] == user_id]
        .sort_values("window_end_date", ascending=False)
        .head(8)
    )
    records = [
        {
            "window_end_date": r["window_end_date"],
            "score": round(float(r["rule_score"]) * 100),
            "recommendation": r["rule_recommendation"],
            "sleep_change_pct": _clean(r.get("sleep_change_pct")),
            "hr_elevation_bpm": _clean(r.get("hr_elevation_bpm")),
            "training_load_ratio": _clean(r.get("training_load_ratio")),
        }
        for _, r in history.iterrows()
    ]

    raw_cols = [
        "user_id", "window_end_date", "sleep_avg_7d", "resting_hr_avg_7d",
        "rmssd_avg_7d", "sleep_efficiency_avg_7d", "steps_avg_7d",
        "active_minutes_avg_7d", "workouts_count", "sleep_change_pct",
        "hr_elevation_bpm", "training_load_ratio", "rule_score",
    ]
    raw_rows = []
    for _, r in history.head(6).iterrows():
        raw_rows.append([
            r[c] if c in ("user_id", "window_end_date")
            else (None if pd.isna(r[c]) else round(float(r[c]), 2))
            for c in raw_cols
        ])

    sources = int(STORE.daily["source"].nunique()) if "source" in STORE.daily.columns else 1

    return {
        "user_id": user_id,
        "display_name": display_name(user_id),
        "gender": _demo("gender"),
        "age_group": _demo("age_group"),
        "sport_type": _demo("sport_type"),
        "window_end_date": row["window_end_date"],
        "score": score,
        "label": score_to_band(rule.score),
        "recommendation": rule.recommendation,
        "ml_confidence": ml_confidence,
        "checks": checks,
        "moved": moved,
        "explanation": explanation,
        "fairness": fairness,
        "records": records,
        "raw": {"columns": raw_cols, "rows": raw_rows},
        "dataset": {
            "users": int(STORE.usable["user_id"].nunique()),
            "records": int(len(STORE.usable)),
            "sources": sources,
        },
    }

@app.get("/")
def health():
    return {"status": "ok", "users": int(STORE.usable["user_id"].nunique())}
