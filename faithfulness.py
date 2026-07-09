# Faithfulness hallucination suppression module.
#
# Checks whether an LLM generated explanation actually reflects the rule engine and the SHAP feature ranking. Produces a faithfulness score in [0, 1].

from __future__ import annotations

from dataclasses import dataclass

FAITHFULNESS_THRESHOLD = 0.60

# Weights for the four sub scores (sum to 1.0)
W_RECOMMENDATION = 0.35
W_DIRECTION = 0.25
W_GROUNDING = 0.20
W_FABRICATION = 0.20

RECOMMENDATION_ALIASES = {
    "Rest Day": ["rest day", "rest", "recover", "take it easy", "day off"],
    "Light Activity": ["light activity", "light", "easy session", "moderate", "gentle"],
    "Intensive Training": ["intensive", "intense", "hard session", "high intensity", "train hard", "well recovered"],
}

# Medical red flags that the wearable model has no basis to claim
FORBIDDEN_CLAIMS = [
    "vo2max", "vo2 max", "blood pressure", "cholesterol", "diagnos", "disease",
    "medication", "cardiac event", "arrhythmia", "covid", "illness", "prescrib",
    "spo2", "blood oxygen", "glucose", "calorie deficit",
]


@dataclass
class FaithfulnessResult:
    text: str            # the text to actually show the user
    score: float         # faithfulness score of the candidate LLM text
    suppressed: bool     # True if LLM text was rejected and fallback used
    breakdown: dict      # check sub scores
    reasons: list[str]   # why points were lost


def _mentions(text: str, phrases: list[str]) -> bool:
    return any(p in text for p in phrases)


def score_faithfulness(candidate: str, evidence: dict) -> FaithfulnessResult:
    text = candidate.lower()
    reasons: list[str] = []

    rec = evidence["recommendation"]
    ranking = evidence["shap_ranking"]
    rule = evidence["rule"]

    # 1. Recommendation agreement
    stated_recs = {
        name for name, aliases in RECOMMENDATION_ALIASES.items() if _mentions(text, aliases)
    }
    if rec in stated_recs and len(stated_recs) == 1:
        s_rec = 1.0
    elif rec in stated_recs:
        s_rec = 0.6            # correct one mentioned but also others, ambiguous
        reasons.append("Explanation mentions multiple conflicting recommendations.")
    elif stated_recs:
        s_rec = 0.0            # states a DIFFERENT recommendation, serious
        reasons.append(f"Explanation states a different recommendation than the model ({stated_recs}).")
    else:
        s_rec = 0.3            # does not clearly state any recommendation
        reasons.append("Explanation does not clearly state the recommendation.")

    # 2. Direction consistency
    # If a top SHAP feature is mentioned, check the text does not call it good when it actually pushed against the recommendation, or bad when it supported it.
    from explain import FEATURE_LABELS
    POS = ["good", "strong", "excellent", "well", "supports", "improv", "healthy", "ready", "optimal", "high hrv"]
    NEG = ["poor", "low", "suppress", "deficit", "elevated", "fatigue", "stress", "risk", "insufficient", "need rest"]
    checked = 0
    consistent = 0
    for c in ranking:
        label = FEATURE_LABELS.get(c.feature, c.feature).split(" (")[0].lower()
        key = label.split()[0]
        if key and key in text:
            checked += 1
            # find a window of text around the mention
            window = text
            says_pos = any(w in window for w in POS)
            says_neg = any(w in window for w in NEG)
            shap_pos = c.shap_value > 0
            # only fail when the text clearly says the opposite (good vs bad)
            if shap_pos and says_neg and not says_pos:
                reasons.append(f"'{label}' described negatively but it supports the recommendation.")
            elif (not shap_pos) and says_pos and not says_neg:
                reasons.append(f"'{label}' described positively but it counts against the recommendation.")
            else:
                consistent += 1
    s_dir = 1.0 if checked == 0 else consistent / checked

    # 3. Grounding: does the text mention the main SHAP ?
    top_feats = ranking[: min(3, len(ranking))]
    if top_feats:
        mentioned = 0
        for c in top_feats:
            label = FEATURE_LABELS.get(c.feature, c.feature).split(" (")[0].lower()
            if any(tok in text for tok in label.split()):
                mentioned += 1
        s_ground = mentioned / len(top_feats)
        if s_ground < 0.5:
            reasons.append("Explanation omits most of the top SHAP drivers.")
    else:
        s_ground = 1.0

    # 4. Fabrication: block medical claims the model has no data for
    s_fab = 1.0
    for claim in FORBIDDEN_CLAIMS:
        if claim in text:
            s_fab = 0.0
            reasons.append(f"Explanation makes an unsupported claim: '{claim}'.")
            break

    score = (W_RECOMMENDATION * s_rec + W_DIRECTION * s_dir
             + W_GROUNDING * s_ground + W_FABRICATION * s_fab)
    breakdown = {
        "recommendation_agreement": round(s_rec, 3),
        "direction_consistency": round(s_dir, 3),
        "grounding_coverage": round(s_ground, 3),
        "fabrication_check": round(s_fab, 3),
    }
    return FaithfulnessResult(text=candidate, score=round(score, 3),
                              suppressed=False, breakdown=breakdown, reasons=reasons)


def gate_llm_output(candidate: str, evidence: dict, fallback_text: str,
                    threshold: float = FAITHFULNESS_THRESHOLD) -> FaithfulnessResult:
    result = score_faithfulness(candidate, evidence)
    if result.score < threshold:
        result.suppressed = True
        result.text = fallback_text
        result.reasons.append(
            f"Faithfulness {result.score:.2f} < {threshold:.2f}: LLM output suppressed, "
            f"showing verified template."
        )
    return result


if __name__ == "__main__":
    # a faithful vs a hallucinated explanation on a real profile.
    import pandas as pd
    from pathlib import Path
    from explain import ExplanationEngine

    df = pd.read_csv(Path(__file__).resolve().parent / "data" / "processed" / "profiles_7day.csv")
    engine = ExplanationEngine()
    row = df.sample(1, random_state=3).iloc[0]
    ev = engine.build_evidence(row)
    faithful = engine.explain(row, "expert").text
    print("Recommendation:", ev["recommendation"])
    print("\n-- Faithful candidate --\n", faithful)
    print("  score:", score_faithfulness(faithful, ev).score)

    hallucinated = ("Your VO2max has dropped and your blood pressure is high, so you should "
                    "rest. Your glucose was 142 today.")
    r = gate_llm_output(hallucinated, ev, fallback_text=faithful)
    print("\n-- Hallucinated candidate --\n", hallucinated)
    print("  score:", r.score, "| suppressed:", r.suppressed)
    print("  reasons:", r.reasons)
    print("  shown to user:\n ", r.text)
