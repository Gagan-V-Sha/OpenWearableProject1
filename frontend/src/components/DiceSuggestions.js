import { useEffect, useState } from "react";
import { getSuggest } from "../api";
import { LABEL_COLORS } from "../format";

const FEATURE_LABELS = {
  sleep_avg_7d: "Sleep",
  training_load_ratio: "Training load",
  workouts_count: "Workouts / week",
  sleep_efficiency_avg_7d: "Sleep efficiency",
};

function fmtFeature(feature, v) {
  if (v == null) return "–";
  if (feature === "sleep_avg_7d") {
    const h = Math.floor(v);
    const m = Math.round((v - h) * 60);
    return `${h}h ${String(m).padStart(2, "0")}m`;
  }
  if (feature === "training_load_ratio") return `${v.toFixed(2)}x`;
  if (feature === "sleep_efficiency_avg_7d") return `${Math.round(v)}%`;
  return `${Math.round(v)}`;
}

// Horizontal 0–100 score meter with a marker at the current score, a marker
// at the predicted score, and a filled band between them.
function ScoreMeter({ fromScore, toScore, fromBand, toBand }) {
  const lo = Math.min(fromScore, toScore);
  const hi = Math.max(fromScore, toScore);
  const color = LABEL_COLORS[toBand] || "var(--accent)";
  return (
    <div className="cf-meter">
      <div className="cf-track">
        <div
          className="cf-fill"
          style={{ left: `${lo}%`, width: `${Math.max(hi - lo, 1)}%`, background: color }}
        />
        <div className="cf-marker" style={{ left: `${fromScore}%` }} />
        <div
          className="cf-marker cf-marker-target"
          style={{ left: `${toScore}%`, borderColor: color }}
        />
      </div>
      <div className="cf-meter-labels">
        <span className="muted">
          {fromScore} {fromBand}
        </span>
        <span style={{ color }}>
          {toScore} {toBand} ({toScore - fromScore >= 0 ? "+" : ""}
          {toScore - fromScore})
        </span>
      </div>
    </div>
  );
}

export default function DiceSuggestions({ userId }) {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!userId) return;
    setResult(null);
    setError(null);
    getSuggest(userId)
      .then(setResult)
      .catch((e) => setError(e.message));
  }, [userId]);

  return (
    <section className="card">
      <div className="section-head">
        <div>
          <h3>✦ Counterfactual Suggestion (DiCE)</h3>
          <p className="muted" style={{ margin: "4px 0 0" }}>
            The smallest actionable changes that would move this user to a better
            recommendation — computed with DiCE and verified by the transparent rule engine.
          </p>
        </div>
        {result && <span className="badge badge-accent">DICE-ML</span>}
      </div>

      {error ? (
        <div className="whatif-message">
          <span className="whatif-bulb">💡</span> Counterfactual suggestions are not available
          on this deployment (the ML stack isn't loaded). The rest of the audit works normally.
        </div>
      ) : !result ? (
        <div className="chart-empty">Searching for counterfactuals…</div>
      ) : result.suggestions.length === 0 ? (
        <div className="whatif-message">
          <span className="whatif-bulb">💡</span> {result.message}
        </div>
      ) : (
        <div className="cf-list">
          {result.suggestions.map((s, i) => (
            <div key={i} className="cf-row">
              <div className="cf-changes">
                <div className="cf-option">Option {i + 1}</div>
                {s.changes.map((c) => (
                  <div key={c.feature} className="cf-chip">
                    <span className="cf-chip-label">
                      {FEATURE_LABELS[c.feature] || c.feature}
                    </span>
                    <span className="cf-chip-values">
                      {fmtFeature(c.feature, c.before)} <span className="cf-arrow">→</span>{" "}
                      <strong>{fmtFeature(c.feature, c.after)}</strong>
                    </span>
                  </div>
                ))}
                <div className="muted small cf-outcome">
                  {s.from_band} → {s.to_band}
                </div>
              </div>
              <ScoreMeter
                fromScore={s.from_score}
                toScore={s.to_score}
                fromBand={s.from_band}
                toBand={s.to_band}
              />
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
