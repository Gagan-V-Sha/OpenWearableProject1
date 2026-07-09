

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from features import FEATURES, LABEL_MAP
from rule_engine import assess as rule_assess

ROOT = Path(__file__).resolve().parent
PROFILE_PATH = ROOT / "data" / "processed" / "profiles_7day.csv"
MODEL_PATH = ROOT / "models" / "xgboost_recovery.json"

INT_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

RECOVERY_BAND = {
    "Rest Day": "Poor",
    "Light Activity": "Moderate",
    "Intensive Training": "Good",
}

ACTIONABLE_FEATURES = [
    "sleep_avg_7d",
    "training_load_ratio",
    "workouts_count",
    "sleep_efficiency_avg_7d",
]

PERMITTED_RANGES = {
    "sleep_avg_7d": [4.0, 10.0],
    "training_load_ratio": [0.5, 1.5],
    "workouts_count": [0.0, 7.0],
    "sleep_efficiency_avg_7d": [70.0, 98.0],
}

IMPROVE_ONLY = ("sleep_avg_7d", "sleep_efficiency_avg_7d")

MIN_DELTA = {
    "sleep_avg_7d": 10.0 / 60.0,
    "training_load_ratio": 0.05,
    "workouts_count": 1.0,
    "sleep_efficiency_avg_7d": 1.0,
}

@dataclass
class FeatureChange:
    feature: str
    before: float
    after: float
    text: str

@dataclass
class CounterfactualSuggestion:
    message: str
    changes: list[FeatureChange]
    from_label: str
    to_label: str
    from_band: str
    to_band: str
    from_score: float
    to_score: float
    cf_features: dict = field(default_factory=dict)

@dataclass
class SuggestionResult:
    user_id: str | None
    window_end_date: str | None
    current_label: str
    current_band: str
    suggestions: list[CounterfactualSuggestion]
    message: str

class SuggestionEngine:

    def __init__(self, model_path: Path = MODEL_PATH, profile_path: Path = PROFILE_PATH):
        import dice_ml

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Run train_models.py first.")
        if not profile_path.exists():
            raise FileNotFoundError(f"Profiles not found at {profile_path}. Run build_profiles.py first.")

        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))

        profiles = pd.read_csv(profile_path)
        reference = profiles.dropna(subset=FEATURES).copy()
        self._medians = reference[FEATURES].median()

        dice_frame = reference[FEATURES].copy()
        dice_frame["recommendation_code"] = reference["rule_recommendation"].map(LABEL_MAP)

        self._dice_data = dice_ml.Data(
            dataframe=dice_frame,
            continuous_features=list(FEATURES),
            outcome_name="recommendation_code",
        )
        self._dice_model = dice_ml.Model(
            model=self.model, backend="sklearn", model_type="classifier"
        )
        self._dice_ml = dice_ml

    def _make_query(self, row: pd.Series | dict) -> pd.DataFrame:
        get = row.get if hasattr(row, "get") else (lambda k, d=None: row.get(k, d))
        values = {f: get(f, np.nan) for f in FEATURES}
        query = pd.DataFrame([values])[FEATURES].astype(float)

        return query.fillna(self._medians)

    @staticmethod
    def _permitted_ranges_for(query: pd.DataFrame) -> dict:

        ranges = {k: list(v) for k, v in PERMITTED_RANGES.items()}
        for f in IMPROVE_ONLY:
            lo, hi = ranges[f]
            current = float(query.iloc[0][f])
            ranges[f] = [min(max(current, lo), hi), hi]
        return ranges

    def _run_dice(self, query: pd.DataFrame, desired_class: int, n_cfs: int) -> pd.DataFrame | None:

        permitted = self._permitted_ranges_for(query)
        for method in ("genetic", "random"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    explainer = self._dice_ml.Dice(self._dice_data, self._dice_model, method=method)
                    result = explainer.generate_counterfactuals(
                        query,
                        total_CFs=n_cfs,
                        desired_class=desired_class,
                        features_to_vary=ACTIONABLE_FEATURES,
                        permitted_range=permitted,
                    )
                cfs = result.cf_examples_list[0].final_cfs_df
                if cfs is not None and len(cfs):
                    return cfs.drop(columns=["recommendation_code"], errors="ignore").astype(float)
            except Exception as e:
                print(f"[suggest] DiCE method '{method}' failed ({e}); trying next.")
        return None

    def _apply_coupling(self, cf: pd.Series, base_row: pd.Series | dict) -> pd.Series:
        get = base_row.get if hasattr(base_row, "get") else (lambda k, d=None: base_row.get(k, d))
        cf = cf.copy()

        cf["workouts_count"] = float(round(cf["workouts_count"]))

        prev_sleep = get("sleep_avg_prev_7d", np.nan)
        if pd.notna(prev_sleep) and prev_sleep > 0:
            cf["sleep_change_pct"] = (cf["sleep_avg_7d"] - prev_sleep) / prev_sleep * 100.0

        prev_active = get("active_minutes_avg_prev_7d", np.nan)
        if pd.notna(prev_active) and prev_active > 0:
            cf["active_minutes_avg_7d"] = cf["training_load_ratio"] * prev_active

        return cf

    @staticmethod
    def _change_text(feature: str, before: float, after: float) -> str:
        if feature == "sleep_avg_7d":
            delta_min = (after - before) * 60.0
            direction = "more" if delta_min > 0 else "fewer"
            return (f"slept about {abs(delta_min):.0f} {direction} minutes per night "
                    f"({before:.1f}h -> {after:.1f}h)")
        if feature == "workouts_count":
            b, a = int(round(before)), int(round(after))
            verb = "reduced" if a < b else "increased"
            return f"{verb} your training sessions from {b} to {a} this week"
        if feature == "training_load_ratio":
            verb = "eased" if after < before else "raised"
            return (f"{verb} your weekly training load from {before:.2f}x to "
                    f"{after:.2f}x of your usual volume")
        if feature == "sleep_efficiency_avg_7d":
            verb = "improved" if after > before else "let"
            if after > before:
                return f"improved your sleep efficiency from {before:.0f}% to {after:.0f}%"
            return f"had your sleep efficiency at {after:.0f}% instead of {before:.0f}%"
        return f"changed {feature} from {before:.2f} to {after:.2f}"

    def _build_suggestion(self, base_query: pd.DataFrame, base_row, cf: pd.Series,
                          from_cls: int, to_cls: int) -> CounterfactualSuggestion | None:
        changes: list[FeatureChange] = []
        for f in ACTIONABLE_FEATURES:
            before = float(base_query.iloc[0][f])
            after = float(cf[f])
            if abs(after - before) >= MIN_DELTA[f]:
                changes.append(FeatureChange(f, before, after, self._change_text(f, before, after)))
        if not changes:
            return None

        from_label, to_label = INT_TO_LABEL[from_cls], INT_TO_LABEL[to_cls]

        base_assess = rule_assess({**base_query.iloc[0].to_dict()})
        cf_assess = rule_assess(cf.to_dict())

        parts = [c.text for c in changes]
        if len(parts) == 1:
            joined = parts[0]
        else:
            joined = ", ".join(parts[:-1]) + (", and " if len(parts) > 2 else " and ") + parts[-1]

        message = (
            f"If you {joined}, your recovery would shift from "
            f"{RECOVERY_BAND[from_label]} ({from_label}) to "
            f"{RECOVERY_BAND[to_label]} ({to_label})."
        )

        return CounterfactualSuggestion(
            message=message,
            changes=changes,
            from_label=from_label,
            to_label=to_label,
            from_band=RECOVERY_BAND[from_label],
            to_band=RECOVERY_BAND[to_label],
            from_score=base_assess.score,
            to_score=cf_assess.score,
            cf_features={f: float(cf[f]) for f in FEATURES},
        )

    @staticmethod
    def _effort(suggestion: CounterfactualSuggestion) -> float:
        scale = {"sleep_avg_7d": 2.0, "training_load_ratio": 0.8,
                 "workouts_count": 4.0, "sleep_efficiency_avg_7d": 15.0}
        return sum(abs(c.after - c.before) / scale[c.feature] for c in suggestion.changes)

    def suggest(self, row: pd.Series | dict, total_cfs: int = 3,
                search_cfs: int = 10) -> SuggestionResult:

        get = row.get if hasattr(row, "get") else (lambda k, d=None: row.get(k, d))
        query = self._make_query(row)
        base_cls = int(self.model.predict(query)[0])
        base_label = INT_TO_LABEL[base_cls]

        result = SuggestionResult(
            user_id=get("user_id"),
            window_end_date=get("window_end_date"),
            current_label=base_label,
            current_band=RECOVERY_BAND[base_label],
            suggestions=[],
            message="",
        )

        if base_cls == LABEL_MAP["Intensive Training"]:
            result.message = ("You are already in the Good recovery band and cleared for "
                              "intensive training - no changes needed.")
            return result

        desired_cls = base_cls + 1
        raw_cfs = self._run_dice(query, desired_cls, search_cfs)
        if raw_cfs is None:
            result.message = ("No feasible counterfactual found within realistic ranges of "
                              "sleep, training load and workout frequency.")
            return result

        suggestions: list[CounterfactualSuggestion] = []
        seen: set[tuple] = set()
        for _, cf in raw_cfs.iterrows():
            cf = self._apply_coupling(cf, row)

            verified_cls = int(self.model.predict(pd.DataFrame([cf])[FEATURES])[0])
            if verified_cls < desired_cls:
                continue
            s = self._build_suggestion(query, row, cf, base_cls, verified_cls)
            if s is None:
                continue

            if s.to_score < s.from_score - 1e-9:
                continue
            key = tuple((c.feature, round(c.after, 1)) for c in s.changes)
            if key in seen:
                continue
            seen.add(key)
            suggestions.append(s)

        suggestions.sort(key=self._effort)
        result.suggestions = suggestions[:total_cfs]
        if result.suggestions:
            result.message = result.suggestions[0].message
        else:
            result.message = ("DiCE found candidate changes, but none survived the "
                              "consistency re-check against the model.")
        return result

def _print_result(res: SuggestionResult) -> None:
    print("=" * 72)
    print(f"User {res.user_id} | window end {res.window_end_date}")
    print(f"Current state: {res.current_band} ({res.current_label})")
    print("=" * 72)
    if not res.suggestions:
        print(res.message)
        print()
        return
    for i, s in enumerate(res.suggestions, 1):
        print(f"Suggestion {i}: {s.message}")
        print(f"  (rule recovery score {s.from_score:.2f} -> {s.to_score:.2f})")
    print()

if __name__ == "__main__":
    engine = SuggestionEngine()
    df = pd.read_csv(PROFILE_PATH)
    usable = df.dropna(subset=FEATURES).reset_index(drop=True)
    preds = engine.model.predict(usable[FEATURES])

    for target in ("Rest Day", "Light Activity"):
        idx = np.where(preds == LABEL_MAP[target])[0]
        if len(idx):
            _print_result(engine.suggest(usable.iloc[idx[0]]))
