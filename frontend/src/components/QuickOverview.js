import { SparkBars } from "./Charts";
import { formatHours, formatSteps, formatSigned } from "../format";

function fmtShort(iso) {
  if (!iso) return "";
  const d = new Date(`${iso}T00:00:00`);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function MetricCard({ icon, title, value, unit, badge, badgeTone, spark, sparkTitles, color }) {
  return (
    <div className="card metric-card">
      <div className="metric-head">
        <span className="metric-icon">{icon}</span>
        <span className="metric-title">{title}</span>
      </div>
      <div className="metric-value">
        {value} {unit && <span className="metric-unit">{unit}</span>}
      </div>
      <div className={`metric-badge tone-${badgeTone}`}>{badge}</div>
      <SparkBars values={spark} titles={sparkTitles} color={color} />
    </div>
  );
}

export default function QuickOverview({ data }) {
  const daily = data.daily.slice(-7);
  const m = data.metrics;

  const sleepDown = (m.sleep.change_pct ?? 0) < 0;
  const hrUp = (m.resting_hr.change_bpm ?? 0) > 0;
  const loadHigh = m.training_load.level === "High";

  const titles = (fmt) =>
    daily.map((d, i) => `${fmtShort(d.date)} · ${fmt(daily[i]) ?? "no data"}`);

  return (
    <section>
      <div className="section-head">
        <h3>Quick Overview</h3>
        <span className="chip chip-accent">{data.window_end_date}</span>
      </div>
      <div className="metric-grid">
        <MetricCard
          icon="🌙"
          title="Sleep"
          value={formatHours(m.sleep.avg_hours_7d)}
          badge={`${formatSigned(m.sleep.change_pct, 0, "%")} vs baseline`}
          badgeTone={sleepDown ? "bad" : "good"}
          spark={daily.map((d) => d.sleep_hours)}
          sparkTitles={titles((d) => (d.sleep_hours != null ? formatHours(d.sleep_hours) : null))}
          color="var(--orange)"
        />
        <MetricCard
          icon="❤️"
          title="Resting heart rate"
          value={m.resting_hr.avg_bpm_7d ?? "–"}
          unit="bpm"
          badge={`${formatSigned(m.resting_hr.change_bpm, 1)} vs baseline`}
          badgeTone={hrUp ? "bad" : "good"}
          spark={daily.map((d) => d.resting_hr)}
          sparkTitles={titles((d) => (d.resting_hr != null ? `${d.resting_hr} bpm` : null))}
          color="var(--violet)"
        />
        <MetricCard
          icon="👟"
          title="Activity"
          value={formatSteps(m.activity.steps_avg_per_day)}
          unit="steps/day"
          badge={`${formatSigned(m.activity.change_pct, 0, "%")} vs prev week`}
          badgeTone="neutral"
          spark={daily.map((d) => d.steps)}
          sparkTitles={titles((d) => (d.steps != null ? `${formatSteps(d.steps)} steps` : null))}
          color="var(--blue)"
        />
        <MetricCard
          icon="🏋️"
          title="Training Load"
          value={m.training_load.active_minutes_7d ?? "–"}
          unit="active min"
          badge={`${m.training_load.level} · ${formatSigned(m.training_load.change_pct, 0, "%")} vs week`}
          badgeTone={loadHigh ? "warn" : "neutral"}
          spark={daily.map((d) => d.active_minutes)}
          sparkTitles={titles((d) =>
            d.active_minutes != null ? `${Math.round(d.active_minutes)} active min` : null
          )}
          color="var(--green)"
        />
      </div>
    </section>
  );
}
