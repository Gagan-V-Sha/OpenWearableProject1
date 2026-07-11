# Fairness Audit Layer.
# After predictions are generated, this module computes two group-fairness
# metrics across each protected attribute (gender, age group):
#
#   SPD  Statistical Parity Difference
#   EOD  Equal Opportunity Difference
#
# If SPD or EOD exceeds its threshold, the group is FLAGGED and the module reweight sample weights that train_models.py can consume in the next training cycle to reduce the disparity.


from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from features import FEATURES, LABEL_MAP

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
PROFILE_PATH = PROCESSED / "profiles_7day.csv"
DEMO_PATH = PROCESSED / "user_demographics.csv"
MODEL_PATH = ROOT / "models" / "xgboost_recovery.json"
REPORT_PATH = PROCESSED / "fairness_report.json"
WEIGHTS_PATH = PROCESSED / "fairness_weights.csv"

PROTECTED_ATTRS = ["gender", "age_group", "sport_type"]
FAVORABLE_LABEL = "Intensive Training"
TRUTH_COLUMN = "rule_recommendation"

SPD_THRESHOLD = 0.10
EOD_THRESHOLD = 0.10
MIN_GROUP_SIZE = 30
INT_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

def load_scored() -> pd.DataFrame:
    df = pd.read_csv(PROFILE_PATH)
    demo = pd.read_csv(DEMO_PATH)
    df = df.merge(demo, on="user_id", how="left")

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    X = df[FEATURES]
    df["pred_recommendation"] = [INT_TO_LABEL[int(p)] for p in model.predict(X)]
    return df

def _rate_favorable(mask: pd.Series, pred: pd.Series) -> float:
    sub = pred[mask]
    return float((sub == FAVORABLE_LABEL).mean()) if len(sub) else float("nan")

def _tpr_favorable(mask: pd.Series, pred: pd.Series, truth: pd.Series) -> float:

    pos = mask & (truth == FAVORABLE_LABEL)
    if pos.sum() == 0:
        return float("nan")
    return float((pred[pos] == FAVORABLE_LABEL).mean())

def audit_attribute(df: pd.DataFrame, attr: str) -> dict:
    groups = [g for g, n in df[attr].value_counts().items()
              if n >= MIN_GROUP_SIZE and str(g).lower() != "unknown"]
    if len(groups) < 2:
        return {"attribute": attr, "status": "skipped",
                "reason": f"fewer than 2 groups with >= {MIN_GROUP_SIZE} samples", "groups": groups}

    ref = df[attr].value_counts().loc[groups].idxmax()
    pred, truth = df["pred_recommendation"], df[TRUTH_COLUMN]

    ref_spd = _rate_favorable(df[attr] == ref, pred)
    ref_eod = _tpr_favorable(df[attr] == ref, pred, truth)

    rows = []
    flagged = False
    for g in groups:
        mask = df[attr] == g
        rate = _rate_favorable(mask, pred)
        tpr = _tpr_favorable(mask, pred, truth)
        spd = rate - ref_spd
        eod = tpr - ref_eod
        breach = (abs(spd) > SPD_THRESHOLD) or (not np.isnan(eod) and abs(eod) > EOD_THRESHOLD)
        flagged = flagged or (g != ref and breach)
        rows.append({
            "group": g, "n": int(mask.sum()),
            "favorable_rate": round(rate, 3),
            "tpr_favorable": None if np.isnan(tpr) else round(tpr, 3),
            "SPD_vs_ref": round(spd, 3),
            "EOD_vs_ref": None if np.isnan(eod) else round(eod, 3),
            "breach": bool(g != ref and breach),
        })
    return {"attribute": attr, "status": "flagged" if flagged else "ok",
            "reference_group": ref, "favorable_label": FAVORABLE_LABEL, "groups": rows}

def kamiran_calders_weights(df: pd.DataFrame, attr: str) -> pd.Series:

    label = df[TRUTH_COLUMN]
    valid = df[attr].notna() & (df[attr].astype(str).str.lower() != "unknown")
    n = valid.sum()
    w = pd.Series(1.0, index=df.index)
    if n == 0:
        return w
    p_group = df.loc[valid, attr].value_counts(normalize=True)
    p_label = label[valid].value_counts(normalize=True)
    for (g, lab), cell in df[valid].groupby([attr, label]):
        observed = len(cell) / n
        expected = p_group.get(g, 0) * p_label.get(lab, 0)
        if observed > 0:
            w.loc[cell.index] = expected / observed
    return w

def run_audit() -> dict:
    df = load_scored()
    print("=" * 60)
    print("FAIRNESS AUDIT")
    print("=" * 60)
    print(f"Favorable outcome: '{FAVORABLE_LABEL}' | EOD truth: '{TRUTH_COLUMN}'")
    print(f"Thresholds: |SPD| > {SPD_THRESHOLD}, |EOD| > {EOD_THRESHOLD}\n")

    report = {"favorable_label": FAVORABLE_LABEL, "truth": TRUTH_COLUMN,
              "spd_threshold": SPD_THRESHOLD, "eod_threshold": EOD_THRESHOLD,
              "attributes": []}
    any_flag = False
    flagged_attrs = []

    for attr in PROTECTED_ATTRS:
        if attr not in df.columns:
            continue
        res = audit_attribute(df, attr)
        report["attributes"].append(res)
        print(f"[{attr}] -> {res['status'].upper()}")
        if res["status"] == "skipped":
            print(f"    skipped: {res['reason']}")
        else:
            print(f"    reference group: {res['reference_group']}")
            for r in res["groups"]:
                flag = "  <-- FLAGGED" if r["breach"] else ""
                print(f"    {r['group']:<10} n={r['n']:<5} "
                      f"favorable_rate={r['favorable_rate']:.3f} "
                      f"SPD={r['SPD_vs_ref']:+.3f} "
                      f"EOD={r['EOD_vs_ref'] if r['EOD_vs_ref'] is not None else 'NA'}{flag}")
            if res["status"] == "flagged":
                any_flag = True
                flagged_attrs.append(attr)
        print()

    if flagged_attrs:
        primary = flagged_attrs[0]
        weights = kamiran_calders_weights(df, primary)
        out = df[["user_id", "window_end_date"]].copy()
        out["fairness_weight"] = weights.values
        out.to_csv(WEIGHTS_PATH, index=False)
        print(f"DISCREPANCY DETECTED on {flagged_attrs}. Emitted reweighing based on "
              f"'{primary}' -> {WEIGHTS_PATH}")
        print("  train_models.py will apply these weights in the next cycle.")
        report["reweighing"] = {"applied_on": primary, "weights_file": str(WEIGHTS_PATH),
                                "weight_range": [round(float(weights.min()), 3),
                                                 round(float(weights.max()), 3)]}
    else:
        if WEIGHTS_PATH.exists():
            WEIGHTS_PATH.unlink()
        print("No fairness discrepancies above threshold. No reweighing needed.")
        report["reweighing"] = None

    report["overall_status"] = "flagged" if any_flag else "ok"
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nFull report written to {REPORT_PATH}")
    return report

if __name__ == "__main__":
    run_audit()
