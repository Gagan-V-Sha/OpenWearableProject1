import { useEffect, useState } from "react";
import { getMonitoring } from "../api";
import { LineChart } from "./Charts";
import ScoreSection from "./ScoreSection";
import QuickOverview from "./QuickOverview";
import TrendsSection from "./TrendsSection";

const RANGES = [30, 60, 90, 120];

export default function MonitoringPage({ users, userId, onSelectUser, data, onNavigate }) {
  const [days, setDays] = useState(90);
  const [history, setHistory] = useState(null);

  useEffect(() => {
    if (!userId) return;
    setHistory(null);
    getMonitoring(userId, days)
      .then((m) => setHistory(m.score_history))
      .catch(() => setHistory([]));
  }, [userId, days]);

  return (
    <div className="page-section">
      <div className="page-head">
        <div className="page-kicker">DATASET EXPLORER</div>
        <h1>Recovery Monitoring</h1>
        <p className="muted">
          Browse recovery insights for any participant in the study dataset. Pick a user and a
          timeline to see how the model read their signals.
        </p>
      </div>

      <div className="card explorer-bar">
        <div className="explorer-users">
          <div className="card-kicker">SELECT USER</div>
          <div className="user-chips">
            {users.map((u) => (
              <button
                key={u.user_id}
                className={u.user_id === userId ? "user-chip selected" : "user-chip"}
                onClick={() => onSelectUser(u.user_id)}
              >
                <span className="user-chip-name">{u.display_name}</span>
                <span className="user-chip-sub">
                  {[u.gender, u.age_group].filter(Boolean).join(" · ") || "—"}
                </span>
              </button>
            ))}
          </div>
        </div>
        <div className="explorer-timeline">
          <div className="card-kicker">TIMELINE</div>
          <div className="segmented">
            {RANGES.map((r) => (
              <button key={r} className={r === days ? "active" : ""} onClick={() => setDays(r)}>
                {r}D
              </button>
            ))}
          </div>
          {data && <span className="chip chip-accent">Ending {data.window_end_date}</span>}
        </div>
      </div>

      {!data ? (
        <div className="loading">Loading participant data…</div>
      ) : (
        <>
          <ScoreSection data={data} />

          <div className="card">
            <div className="section-head">
              <div>
                <div className="trend-head-title">Recovery graph</div>
                <div className="muted small">Rule-engine recovery score per 7-day window</div>
              </div>
              <span className="muted small">
                {history ? `${history.length} windows` : "Loading…"}
              </span>
            </div>
            <LineChart
              points={(history || []).map((h) => ({ date: h.date, value: h.score }))}
              xLabel="Date"
              yLabel="Recovery score (0–100)"
            />
          </div>

          <QuickOverview data={data} />
          <TrendsSection data={data} />

          <div className="card reason-cta">
            <div>
              <h3>✦ Want the reasoning behind this?</h3>
              <p className="muted">
                See the exact rules, factor weights and fairness checks behind{" "}
                {data.display_name}'s score.
              </p>
            </div>
            <button className="btn-cta" onClick={() => onNavigate("audit")}>
              Know more about the score →
            </button>
          </div>
        </>
      )}
    </div>
  );
}
