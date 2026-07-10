import { Gauge } from "./Charts";
import { LABEL_COLORS } from "../format";

const ADVICE = {
  "Rest Day":
    "Take it easy today — light walking or stretching is great, but skip the hard intervals. They'd set your recovery back further.",
  "Light Activity":
    "Keep it easy — a light jog, spin or mobility session is ideal today. Save the hard intervals for a better-recovered day.",
  "Intensive Training":
    "Green light — your body is ready for a demanding session today. Push the intervals while your recovery is strong.",
};

const REC_ICON = {
  "Rest Day": "🛌",
  "Light Activity": "🚶",
  "Intensive Training": "⚡",
};

export default function ScoreSection({ data }) {
  const color = LABEL_COLORS[data.label];
  return (
    <div className="score-grid">
      <section className="card score-card">
        <Gauge score={data.score} label={data.label} caption="Recovery" />
        <div className="score-text">
          <span className="pill" style={{ color, borderColor: color }}>
            ● {data.label}
          </span>
          <p>{data.narrative}</p>
        </div>
      </section>

      <section className="card rec-card">
        <div className="card-kicker">TODAY'S RECOMMENDATION</div>
        <div className="rec-title">
          <span className="rec-icon">{REC_ICON[data.recommendation] || "✦"}</span>
          <h2>{data.recommendation}</h2>
        </div>
        <p className="rec-advice">{ADVICE[data.recommendation]}</p>
        <p className="rec-basis">{data.explanation}</p>
        {data.ml?.anomaly && (
          <div className="ml-line ml-warn">
            ⚠ Unusual physiological pattern detected (Isolation Forest)
          </div>
        )}
        {data.ml?.recommendation && (
          <div className="ml-line">
            ML check (XGBoost): {data.ml.recommendation} —{" "}
            {data.ml.agrees ? "agrees with the rules" : "differs from the rules"}
          </div>
        )}
        <div className="card-footnote">Based on your last 7 days vs your baseline week</div>
      </section>
    </div>
  );
}
