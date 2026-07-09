# Rule engine. Single source of truth for the recovery rules, used for build_profiles.py and the explanation generator.


from dataclasses import dataclass, field
import pandas as pd

# Recommendation thresholds on the recovery score [0, 1]
REST_THRESHOLD = 0.40
INTENSIVE_THRESHOLD = 0.60

BASE_SCORE = 0.50


@dataclass
class RuleContribution:
    signal: str        # feature/signal name
    value: float       # observed value
    delta: float       # change applied to the recovery score
    message: str       # human-readable justification
    citation: str      # scientific grounding


@dataclass
class RuleAssessment:
    score: float
    recommendation: str
    contributions: list[RuleContribution] = field(default_factory=list)

    def fired(self) -> list[RuleContribution]:
        return [c for c in self.contributions if c.delta != 0.0]


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_to_recommendation(score: float) -> str:
    if score < REST_THRESHOLD:
        return "Rest Day"
    if score < INTENSIVE_THRESHOLD:
        return "Light Activity"
    return "Intensive Training"


def assess(row: pd.Series | dict) -> RuleAssessment:
    get = row.get if hasattr(row, "get") else (lambda k, d=None: row.get(k, d))

    sleep_change = get("sleep_change_pct")
    hr_change = get("hr_elevation_bpm")
    load = get("training_load_ratio")
    rmssd = get("rmssd_avg_7d")
    efficiency = get("sleep_efficiency_avg_7d")
    workouts = get("workouts_count")

    score = BASE_SCORE
    contribs: list[RuleContribution] = []

    def add(signal, value, delta, message, citation):
        nonlocal score
        score += delta
        contribs.append(RuleContribution(signal, value, delta, message, citation))

    # Signal 1: HRV / RMSSD — strongest recovery indicator.
    if pd.notna(rmssd):
        if rmssd > 60.0:
            add("rmssd_avg_7d", rmssd, +0.15, "HRV is excellent, indicating strong autonomic recovery.", "Plews et al. 2013")
        elif rmssd >= 40.0:
            add("rmssd_avg_7d", rmssd, +0.05, "HRV is in a healthy range.", "Plews et al. 2013")
        elif rmssd >= 25.0:
            add("rmssd_avg_7d", rmssd, -0.10, "HRV is suppressed, a sign of incomplete recovery.", "Plews et al. 2013")
        else:
            add("rmssd_avg_7d", rmssd, -0.20, "HRV is very suppressed, a strong signal to rest.", "Plews et al. 2013")

    # Signal 2: Sleep change vs baseline 
    if pd.notna(sleep_change):
        if sleep_change > 5.0:
            add("sleep_change_pct", sleep_change, +0.08, "Sleeping more than your baseline supports extra recovery.", "Fullagar et al. 2015")
        elif sleep_change < -10.0:
            add("sleep_change_pct", sleep_change, -0.20, "A large sleep deficit versus baseline impairs recovery.", "Fullagar et al. 2015")
        elif sleep_change < -5.0:
            add("sleep_change_pct", sleep_change, -0.12, "A moderate sleep deficit versus baseline slows recovery.", "Fullagar et al. 2015")

    # Signal 3: Resting HR elevation vs baseline
    if pd.notna(hr_change):
        if hr_change < -2.0:
            add("hr_elevation_bpm", hr_change, +0.08, "Resting heart rate has dropped, a sign of improving fitness.", "Buchheit 2014")
        elif hr_change <= 2.0:
            pass  # neutral range, no contribution
        elif hr_change <= 4.0:
            add("hr_elevation_bpm", hr_change, -0.10, "Resting heart rate is elevated, suggesting stress or fatigue.", "Buchheit 2014")
        else:
            add("hr_elevation_bpm", hr_change, -0.18, "Resting heart rate is sharply elevated, a marker of overreaching.", "Buchheit 2014")

    # Signal 4: Acute:Chronic Workload Ratio
    if pd.notna(load):
        if load < 0.8:
            add("training_load_ratio", load, -0.05, "Training load is below your baseline (possible deconditioning).", "Gabbett 2016")
        elif load <= 1.3:
            add("training_load_ratio", load, +0.10, "Training load is in the optimal 'sweet spot' zone.", "Gabbett 2016")
        elif load <= 1.5:
            add("training_load_ratio", load, -0.08, "Training load is in the caution zone (elevated injury risk).", "Gabbett 2016")
        else:
            add("training_load_ratio", load, -0.20, "Training load is in the danger zone (high injury risk).", "Gabbett 2016")

    # Signal 5: Sleep efficiency
    if pd.notna(efficiency):
        if efficiency > 85.0:
            add("sleep_efficiency_avg_7d", efficiency, +0.07, "Sleep efficiency is clinically good.", "Ohayon et al. 2017")
        elif efficiency < 75.0:
            add("sleep_efficiency_avg_7d", efficiency, -0.10, "Sleep efficiency is clinically poor.", "Ohayon et al. 2017")

    # Signal 6: Workout frequency
    if pd.notna(workouts):
        if workouts > 5:
            add("workouts_count", workouts, -0.08, "No rest days in the last week raises overtraining risk.", "Kellmann 2010")
        elif workouts >= 3:
            add("workouts_count", workouts, +0.05, "Training frequency is appropriate.", "Kellmann 2010")

    score = _clamp(score)
    return RuleAssessment(score=score, recommendation=score_to_recommendation(score), contributions=contribs)
