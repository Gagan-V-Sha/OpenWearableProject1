import { useEffect, useState } from "react";
import { getAudit, getMonitoring } from "../api";
import { LABEL_COLORS } from "../format";
import { LineChart } from "./Charts";
import DiceSuggestions from "./DiceSuggestions";

function ImpactBar({ impact, max }) {
  const pct = Math.min(100, (Math.abs(impact) / (max || 1)) * 100);
  const positive = impact >= 0;
  return (
    <div className="impact-track">
      <div
        className={positive ? "impact-fill pos" : "impact-fill neg"}
        style={{ width: `${Math.max(4, pct)}%` }}
      />
    </div>
  );
}

function fmtPoints(points) {
  if (points == null || Number.isNaN(points)) return "";
  return points > 0 ? `+${points}` : `${points}`;
}

/** Cutoffs + score points — same logic as rule_engine.py (for the Audit legend). */
const RULE_CUTOFFS = [
  {
    name: "Resting HR change",
    lines: ["< -2 bpm -> +8", "-2 to +2 -> 0", "+2 to +4 -> -10", "> +4 -> -18"],
  },
  {
    name: "Sleep change",
    lines: ["> +5% -> +8", "-5% to -10% -> -12", "< -10% -> -20"],
  },
  {
    name: "Training load",
    lines: ["< 0.8x -> -5", "0.8-1.3x -> +10", "1.3-1.5x -> -8", "> 1.5x -> -20"],
  },
  {
    name: "Sleep efficiency",
    lines: ["> 85% -> +7", "< 75% -> -10"],
  },
  {
    name: "Workouts / week",
    lines: ["3-5 -> +5", "> 5 -> -8"],
  },
  {
    name: "HRV (RMSSD)",
    lines: ["> 60 -> +15", "40-60 -> +5", "25-40 -> -10", "< 25 -> -20"],
  },
];

function FairnessCard({ attr, spdThreshold, eodThreshold }) {
  if (attr.status !== "ok") {
    return (
      <div className="card fairness-card">
        <div className="fairness-head">
          <span>{attr.attribute.replace("_", " ")}</span>
          <span className="badge badge-muted">Skipped</span>
        </div>
        <p className="muted small">{attr.reason || "Not enough groups to audit."}</p>
      </div>
    );
  }
  const groups = attr.groups.filter((g) => g.group !== attr.reference_group);
  const anyBreach = attr.groups.some((g) => g.breach);
  return (
    <div className="card fairness-card">
      <div className="fairness-head">
        <span>{attr.attribute.replace("_", " ")}</span>
        <span className={anyBreach ? "badge badge-fail" : "badge badge-pass"}>
          {anyBreach ? "Flagged" : "Within limits"}
        </span>
      </div>
      {groups.map((g) => (
        <div key={g.group} className="fairness-group">
          <div className="muted small">
            {g.group} vs {attr.reference_group} (n={g.n})
          </div>
          {[
            ["SPD", g.SPD_vs_ref, spdThreshold],
            ["EOD", g.EOD_vs_ref, eodThreshold],
          ].map(([name, value, threshold]) => {
            const num = value == null || Number.isNaN(Number(value)) ? null : Number(value);
            const abs = num == null ? 0 : Math.abs(num);
            const scaleMax = (threshold || 0.1) * 2;
            return (
              <div key={name} className="fairness-row">
                <span className="fairness-metric">{name}</span>
                <div className="fairness-bar-wrap">
                  <div className="fairness-track">
                    <div
                      className={
                        num != null && abs > threshold ? "fairness-fill bad" : "fairness-fill"
                      }
                      style={{
                        width: `${Math.min(100, (abs / scaleMax) * 100)}%`,
                      }}
                    />
                    <div className="fairness-threshold" title={`limit ${threshold}`} />
                  </div>
                  <div className="fairness-scale">
                    <span>0</span>
                    <span className="fairness-scale-mid">{Number(threshold).toFixed(1)} limit</span>
                    <span>{scaleMax.toFixed(1)}</span>
                  </div>
                </div>
                <span className="fairness-value">
                  {num == null ? "n/a" : num.toFixed(2)}
                </span>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

export default function AuditPage({ userId, data }) {
  const [audit, setAudit] = useState(null);
  const [history, setHistory] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!userId) return;
    setAudit(null);
    setHistory(null);
    getAudit(userId)
      .then(setAudit)
      .catch((e) => setError(e.message));
    getMonitoring(userId, 30)
      .then((m) => setHistory(m.score_history))
      .catch(() => setHistory([]));
  }, [userId]);

  if (error) return <div className="loading">Couldn't load the audit: {error}</div>;
  if (!audit) return <div className="loading">Running the audit…</div>;

  const color = LABEL_COLORS[audit.label];
  const maxImpact = Math.max(...audit.moved.items.map((i) => Math.abs(i.impact)), 0.01);
  const kicker = [
    "HOW THE MODEL DECIDED",
    audit.gender,
    audit.age_group,
    audit.window_end_date,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <div className="page-section">
      <div className="page-head">
        <div className="page-kicker">{kicker}</div>
        <div className="audit-title-row">
          <h1>Decision &amp; Fairness Audit</h1>
          <span className="pill" style={{ color, borderColor: color }}>
            ● {audit.display_name} · {audit.score} {audit.label}
          </span>
        </div>
      </div>

      <div className="audit-grid">
        <div className="card">
          <h3>⚖ Rule engine</h3>
          <p className="muted small">
            Numbers in brackets = this athlete&apos;s measured value. Score +/-N = points
            added to the starting 50. Pass adds points; Flag removes them.
          </p>
          {(audit.checks || []).map((c, i) => {
            const points =
              c.points != null
                ? c.points
                : Math.round(Number(c.delta || 0) * 100);
            return (
              <div key={i} className="check-row">
                <div>
                  <div className="check-name">
                    {c.signal} <span className="muted">({c.value})</span>
                  </div>
                  <div className="muted small">
                    {c.message} <span className="check-cite">[{c.citation}]</span>
                  </div>
                </div>
                <div className="check-badges">
                  <span
                    className={
                      points >= 0 ? "badge badge-points pos" : "badge badge-points neg"
                    }
                  >
                    Score {fmtPoints(points)}
                  </span>
                  <span className={c.passed ? "badge badge-pass" : "badge badge-fail"}>
                    {c.passed ? "Pass" : "Flag"}
                  </span>
                </div>
              </div>
            );
          })}
          <div className="score-walk">
            <span className="muted small">How {audit.score} was calculated:</span>
            <div className="score-walk-math">
              <span>50</span>
              {(audit.checks || []).map((c, i) => {
                const points =
                  c.points != null
                    ? c.points
                    : Math.round(Number(c.delta || 0) * 100);
                const sign = points >= 0 ? "+" : "-";
                return (
                  <span key={i}>
                    {" "}
                    {sign} {Math.abs(points)}
                  </span>
                );
              })}
              <span>
                {" "}
                = <strong>{audit.score}</strong> ({audit.label})
              </span>
            </div>
          </div>
          <details className="cutoff-details">
            <summary>Show all cutoffs &amp; points</summary>
            <div className="cutoff-grid">
              {RULE_CUTOFFS.map((r) => (
                <div key={r.name} className="cutoff-card">
                  <div className="cutoff-name">{r.name}</div>
                  {r.lines.map((line) => (
                    <div key={line} className="muted small">
                      {line}
                    </div>
                  ))}
                </div>
              ))}
            </div>
            <p className="muted small" style={{ marginTop: 10 }}>
              Start at 50. Below 40 = Rest · 40–60 = Light · 60+ = Intensive.
            </p>
          </details>
        </div>

        <div className="card">
          <h3>◆ What moved the score</h3>
          <p className="muted small">
            {audit.moved.source === "shap"
              ? "Each factor's SHAP contribution. Red lowered recovery, green lifted it."
              : "Each rule's score contribution. Red lowered recovery, green lifted it."}
          </p>
          {audit.moved.items.map((item, i) => (
            <div key={i} className="impact-row">
              <span className="impact-label">
                {item.label} <span className="muted">({item.value})</span>
              </span>
              <ImpactBar impact={item.impact} max={maxImpact} />
              <span className={item.impact >= 0 ? "impact-value pos" : "impact-value neg"}>
                {item.impact >= 0 ? "+" : ""}
                {item.impact.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="section-head">
          <h3>● Explanation in plain words</h3>
          <span className="badge badge-accent">
            {audit.explanation.faithfulness_score != null
              ? `FAITHFULNESS ${audit.explanation.faithfulness_score.toFixed(2)}`
              : audit.explanation.source.toUpperCase()}
          </span>
        </div>
        <p className="rec-advice">{audit.explanation.text}</p>
      </div>

      <DiceSuggestions userId={userId} />

      <div className="card">
        <div className="section-head">
          <h3>Recovery over time</h3>
          <span className="muted small">Last 30 days</span>
        </div>
        {history === null ? (
          <div className="chart-empty">Loading recovery history…</div>
        ) : (
          <LineChart
            points={history.map((h) => ({ date: h.date, value: h.score }))}
            xLabel="Date"
            yLabel="Recovery score (0–100)"
          />
        )}
      </div>

      <div>
        <div className="section-head">
          <h3>⚖ Fairness audit</h3>
          <span className="muted small">
            SPD and EOD must stay under {audit.fairness?.spd_threshold ?? 0.1}. Anything over is
            flagged and re-weighted in the next training cycle.
          </span>
        </div>
        {audit.fairness ? (
          <div className="fairness-grid">
            {(audit.fairness.attributes || []).map((attr) => (
              <FairnessCard
                key={attr.attribute}
                attr={attr}
                spdThreshold={audit.fairness.spd_threshold}
                eodThreshold={audit.fairness.eod_threshold}
              />
            ))}
          </div>
        ) : (
          <div className="card">
            <p className="muted" style={{ margin: 0 }}>
              No fairness report has been generated for this dataset yet. Run{" "}
              <code>python fairness_audit.py</code> in the backend to compute SPD and EOD
              across gender, age group and sport type — the results will appear here.
            </p>
          </div>
        )}
      </div>

      <div className="card">
        <h3>▤ Records used for {audit.display_name}</h3>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>window_end_date</th>
                <th>recovery_score</th>
                <th>recommendation</th>
                <th>sleep_change_pct</th>
                <th>hr_elevation_bpm</th>
                <th>training_load_ratio</th>
              </tr>
            </thead>
            <tbody>
              {audit.records.map((r, i) => (
                <tr key={i}>
                  <td>{String(i + 1).padStart(2, "0")}</td>
                  <td>{r.window_end_date}</td>
                  <td>{r.score}</td>
                  <td>{r.recommendation}</td>
                  <td>{r.sleep_change_pct ?? "–"}</td>
                  <td>{r.hr_elevation_bpm ?? "–"}</td>
                  <td>{r.training_load_ratio ?? "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <h3>🗄 Dataset used by model</h3>
        <div className="dataset-stats">
          <div className="stat-box">
            <div className="muted small">Users</div>
            <div className="stat-value">{audit.dataset.users}</div>
          </div>
          <div className="stat-box">
            <div className="muted small">Records</div>
            <div className="stat-value">{audit.dataset.records.toLocaleString()}</div>
          </div>
          <div className="stat-box">
            <div className="muted small">Sources</div>
            <div className="stat-value">{audit.dataset.sources}</div>
          </div>
        </div>
        <div className="section-head" style={{ marginTop: 18 }}>
          <div className="trend-head-title">Raw data</div>
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th>
                {audit.raw.columns.map((c) => (
                  <th key={c}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {audit.raw.rows.map((row, i) => (
                <tr key={i}>
                  <td>{String(i + 1).padStart(2, "0")}</td>
                  {row.map((v, j) => (
                    <td key={j}>{v ?? "–"}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
