

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURES
from explain import ExplanationEngine

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
PROFILE_PATH = PROCESSED / "profiles_7day.csv"
ISO_PATH = ROOT / "models" / "isolation_forest.pkl"

def _load_iso():
    if not ISO_PATH.exists():
        return None
    with open(ISO_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle

def respond(row: pd.Series, expertise: str = "expert", use_llm: bool = False) -> dict:
    explainer = ExplanationEngine()
    iso = _load_iso()

    exp = explainer.explain(row, expertise=expertise, use_llm=use_llm)

    anomaly = None
    if iso is not None:
        Xa = pd.DataFrame([{f: row.get(f, np.nan) for f in iso["features"]}])[iso["features"]]
        anomaly = int(iso["model"].predict(Xa)[0]) == -1

    return {"explanation": exp, "anomaly": anomaly}

def _print_response(row: pd.Series, expertise: str, use_llm: bool):
    r = respond(row, expertise=expertise, use_llm=use_llm)
    exp = r["explanation"]
    print("=" * 66)
    print(f"User {row['user_id']} | window end {row['window_end_date']} | audience: {expertise}")
    print("=" * 66)
    print(f"RECOMMENDATION: {exp.recommendation}  (ML confidence {exp.ml_confidence:.0%})")
    if r["anomaly"] is not None:
        print(f"ANOMALY FLAG:   {'YES - unusual physiology, review advised' if r['anomaly'] else 'no'}")
    print(f"EXPLANATION [{exp.source}"
          + (f", faithfulness {exp.faithfulness_score:.2f}" if exp.faithfulness_score is not None else "")
          + "]:")
    print("  " + exp.text.replace("\n", "\n  "))
    print()

if __name__ == "__main__":
    import sys

    use_llm = "--llm" in sys.argv
    df = pd.read_csv(PROFILE_PATH)
    from features import LABEL_MAP
    eng = ExplanationEngine()
    preds = eng.model.predict(df[FEATURES])
    for target_label in ["Rest Day", "Intensive Training"]:
        idx = np.where(preds == LABEL_MAP[target_label])[0]
        if len(idx):
            _print_response(df.iloc[idx[0]], expertise="novice", use_llm=use_llm)
            _print_response(df.iloc[idx[0]], expertise="expert", use_llm=use_llm)
