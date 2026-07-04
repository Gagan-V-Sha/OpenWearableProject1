# Explanation Generator.
# Turns the ML prediction + SHAP feature attributions + transparent rule engine reasoning into natural language, adapted to the user's expertise level (novice / expert).

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from dotenv import load_dotenv

load_dotenv()

from features import FEATURES, LABEL_MAP
from rule_engine import assess as rule_assess, RuleAssessment

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "xgboost_recovery.json"

INT_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# Model features names
FEATURE_LABELS = {
    "sleep_change_pct": "sleep change vs. baseline",
    "hr_elevation_bpm": "resting heart-rate change",
    "training_load_ratio": "training load (acute:chronic)",
    "sleep_avg_7d": "average sleep",
    "resting_hr_avg_7d": "average resting heart rate",
    "steps_avg_7d": "average daily steps",
    "active_minutes_avg_7d": "average active minutes",
    "rmssd_avg_7d": "heart-rate variability (HRV)",
    "sleep_efficiency_avg_7d": "sleep efficiency",
    "workouts_count": "workouts this week",
    "rmssd_missing": "HRV data availability",
}

EXPERTISE_LEVELS = ("novice", "expert")


@dataclass
class ShapContribution:
    feature: str
    value: float
    shap_value: float

    @property
    def direction(self) -> str:
        if self.shap_value > 0:
            return "supports"
        if self.shap_value < 0:
            return "counts against"
        return "neutral for"


@dataclass
class Explanation:
    recommendation: str
    ml_confidence: float
    rule: RuleAssessment
    shap_ranking: list[ShapContribution]
    text: str
    expertise: str
    source: str = "template"
    faithfulness_score: float | None = None
    suppressed: bool = False


class ExplanationEngine:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run train_models.py first."
            )
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))
        self.explainer = shap.TreeExplainer(self.model)

    # core numeric evidence
    def _predict(self, X: pd.DataFrame) -> tuple[int, float]:
        proba = self.model.predict_proba(X)[0]
        cls = int(np.argmax(proba))
        return cls, float(proba[cls])

    def _shap_for_class(self, X: pd.DataFrame, cls: int) -> np.ndarray:
        sv = self.explainer.shap_values(X)
        if isinstance(sv, list):          # list of (n, features) per class
            return np.asarray(sv[cls])[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:                  # (n, features, classes)
            return sv[0, :, cls]
        return sv[0]                      # (n, features)

    def _rank_shap(self, row: pd.Series, shap_vals: np.ndarray, top_k: int) -> list[ShapContribution]:
        contribs = [
            ShapContribution(f, float(row.get(f, np.nan)), float(s))
            for f, s in zip(FEATURES, shap_vals)
        ]
        contribs.sort(key=lambda c: -abs(c.shap_value))
        return contribs[:top_k]

    # public API
    def build_evidence(self, row: pd.Series, top_k: int = 4) -> dict:
        X = pd.DataFrame([{f: row.get(f, np.nan) for f in FEATURES}])[FEATURES]
        cls, conf = self._predict(X)
        shap_vals = self._shap_for_class(X, cls)
        ranking = self._rank_shap(row, shap_vals, top_k)
        rule = rule_assess(row)
        return {
            "recommendation": INT_TO_LABEL[cls],
            "ml_confidence": conf,
            "rule": rule,
            "shap_ranking": ranking,
        }

    def explain(self, row: pd.Series, expertise: str = "novice",
                use_llm: bool = False, top_k: int = 4) -> Explanation:
        if expertise not in EXPERTISE_LEVELS:
            raise ValueError(f"expertise must be one of {EXPERTISE_LEVELS}")

        ev = self.build_evidence(row, top_k=top_k)
        template_text = render_template(ev, expertise)

        exp = Explanation(
            recommendation=ev["recommendation"],
            ml_confidence=ev["ml_confidence"],
            rule=ev["rule"],
            shap_ranking=ev["shap_ranking"],
            text=template_text,
            expertise=expertise,
        )

        if use_llm:
            from faithfulness import gate_llm_output
            llm_text = _try_llm(ev, expertise)
            if llm_text:
                result = gate_llm_output(llm_text, ev, fallback_text=template_text)
                exp.text = result.text
                exp.source = "llm" if not result.suppressed else "fallback"
                exp.faithfulness_score = result.score
                exp.suppressed = result.suppressed
        return exp


# deterministic template
def _fmt(feature: str, value: float) -> str:
    if pd.isna(value):
        return f"{FEATURE_LABELS.get(feature, feature)} (not available)"
    units = {
        "sleep_change_pct": f"{value:+.0f}%",
        "hr_elevation_bpm": f"{value:+.1f} bpm",
        "training_load_ratio": f"{value:.2f}x",
        "sleep_avg_7d": f"{value:.1f} h",
        "resting_hr_avg_7d": f"{value:.0f} bpm",
        "rmssd_avg_7d": f"{value:.0f} ms",
        "sleep_efficiency_avg_7d": f"{value:.0f}%",
        "workouts_count": f"{value:.0f}",
    }
    return f"{FEATURE_LABELS.get(feature, feature)} = {units.get(feature, f'{value:.1f}')}"


def render_template(ev: dict, expertise: str) -> str:
    rec = ev["recommendation"]
    conf = ev["ml_confidence"]
    rule: RuleAssessment = ev["rule"]
    ranking: list[ShapContribution] = ev["shap_ranking"]
    fired = rule.fired()

    # Surface the signals that actually push TOWARD the recommendation: negative contributions justify resting, positive ones justify training.
    if rec == "Rest Day":
        aligned = [c for c in fired if c.delta < 0]
    elif rec == "Intensive Training":
        aligned = [c for c in fired if c.delta > 0]
    else:
        aligned = list(fired)
    aligned = sorted(aligned, key=lambda c: -abs(c.delta)) or fired

    if expertise == "novice":
        lead = {
            "Rest Day": "Your body looks like it needs a rest today.",
            "Light Activity": "A light, easy session is the sweet spot for you today.",
            "Intensive Training": "You look well recovered and ready for a hard session.",
        }[rec]
        reasons = [c.message.rstrip(".") for c in aligned[:2]]
        why = (" Mainly because " + "; and ".join(r.lower() for r in reasons) + ".") if reasons else ""
        return lead + why

    # expert
    shap_lines = [
        f"  - {FEATURE_LABELS.get(c.feature, c.feature)} ({_fmt(c.feature, c.value)}): "
        f"SHAP {c.shap_value:+.3f}, {c.direction} this recommendation"
        for c in ranking
    ]
    rule_lines = [
        f"  - {c.signal}: {c.message} [score {c.delta:+.2f}, {c.citation}]"
        for c in fired
    ]
    return (
        f"Recommendation: {rec} | ML confidence {conf:.1%} | "
        f"rule recovery score {rule.score:.2f} (rule says '{rule.recommendation}').\n"
        f"Top SHAP feature attributions:\n" + "\n".join(shap_lines) + "\n"
        f"Rule-engine contributions:\n" + ("\n".join(rule_lines) if rule_lines else "  (no rules fired)")
    )


# LLM phrase generation (Google Gemini)
def _try_llm(ev: dict, expertise: str) -> str | None:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        rule: RuleAssessment = ev["rule"]
        facts = {
            "recommendation": ev["recommendation"],
            "model_confidence": round(ev["ml_confidence"], 3),
            "rule_recovery_score": round(rule.score, 3),
            "top_shap_features": [
                {"feature": c.feature, "value": None if pd.isna(c.value) else round(c.value, 2),
                 "shap": round(c.shap_value, 3)} for c in ev["shap_ranking"]
            ],
            "rules_fired": [
                {"signal": c.signal, "message": c.message, "delta": c.delta} for c in rule.fired()
            ],
        }
        style = {
            "novice": "Use plain, encouraging language. No jargon. 2 short sentences.",
            "expert": "Be precise and technical. Reference SHAP magnitudes and rule deltas.",
        }[expertise]
        prompt = (
            "You explain a wearable recovery recommendation. Use ONLY the facts in the JSON. "
            "Do not invent numbers, features, or medical claims not present. "
            f"Audience: {expertise}. {style}\n\nFACTS:\n{facts}"
        )
        resp = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=prompt,
        )
        return (resp.text or "").strip() or None
    except Exception as e:  # network/quota/library issues fallback 
        print(f"[explain] Gemini unavailable ({e}); using template.")
        return None


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "processed" / "profiles_7day.csv")
    engine = ExplanationEngine()
    sample = df.sample(1, random_state=7).iloc[0]
    print(f"User: {sample['user_id']}  window end: {sample['window_end_date']}\n")
    for lvl in EXPERTISE_LEVELS:
        exp = engine.explain(sample, expertise=lvl, use_llm=False)
        print(f"===== {lvl.upper()} =====")
        print(exp.text)
        print()
