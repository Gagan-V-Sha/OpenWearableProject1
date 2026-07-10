import { useState } from "react";
import { BarChart, LineChart } from "./Charts";
import { formatHours, formatSteps } from "../format";

const RANGES = [3, 7, 14, 30];

function avg(values) {
  const v = values.filter((x) => x != null);
  if (v.length === 0) return null;
  return v.reduce((a, b) => a + b, 0) / v.length;
}

function fmtShort(iso) {
  if (!iso) return "";
  const d = new Date(`${iso}T00:00:00`);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function TrendCard({ title, avgLabel, data, color, format }) {
  return (
    <div className="card trend-card">
      <div className="trend-head">
        <span>{title}</span>
        <span className="muted">avg {avgLabel}</span>
      </div>
      <BarChart data={data} color={color} format={format} />
      <div className="trend-axis">
        <span>{data.length > 0 ? fmtShort(data[0].date) : ""}</span>
        <span>{data.length > 0 ? fmtShort(data[data.length - 1].date) : ""}</span>
      </div>
    </div>
  );
}

export default function TrendsSection({ data }) {
  const [range, setRange] = useState(7);
  const [hrRange, setHrRange] = useState(14);

  const slice = data.daily.slice(-range);
  const hrSlice = data.daily.slice(-hrRange);
  const hrVals = hrSlice.filter((d) => d.resting_hr != null).map((d) => d.resting_hr);

  return (
    <section>
      <div className="section-head">
        <h3>{range}-day trends</h3>
        <div className="segmented">
          {RANGES.map((r) => (
            <button
              key={r}
              className={r === range ? "active" : ""}
              onClick={() => setRange(r)}
            >
              {r}D
            </button>
          ))}
        </div>
      </div>

      <div className="trend-grid">
        <TrendCard
          title="Sleep"
          avgLabel={formatHours(avg(slice.map((d) => d.sleep_hours)))}
          data={slice.map((d) => ({ date: d.date, value: d.sleep_hours }))}
          color="var(--orange)"
          format={formatHours}
        />
        <TrendCard
          title="Activity"
          avgLabel={formatSteps(avg(slice.map((d) => d.steps)))}
          data={slice.map((d) => ({ date: d.date, value: d.steps }))}
          color="var(--blue)"
          format={(v) => `${formatSteps(v)} steps`}
        />
        <TrendCard
          title="Training load"
          avgLabel={`${Math.round(avg(slice.map((d) => d.active_minutes)) ?? 0)} min`}
          data={slice.map((d) => ({ date: d.date, value: d.active_minutes }))}
          color="var(--green)"
          format={(v) => `${Math.round(v)} active min`}
        />
      </div>

      <div className="card hr-card">
        <div className="section-head">
          <div>
            <div className="trend-head-title">Resting heart rate</div>
            <div className="muted small">
              Last {hrRange} days
              {hrVals.length > 0 && ` · avg ${Math.round(avg(hrVals))} bpm`}
            </div>
          </div>
          <div className="segmented">
            {[7, 14, 30].map((r) => (
              <button
                key={r}
                className={r === hrRange ? "active" : ""}
                onClick={() => setHrRange(r)}
              >
                {r}D
              </button>
            ))}
          </div>
        </div>
        <LineChart
          points={hrSlice.map((d) => ({ date: d.date, value: d.resting_hr }))}
          unit="bpm"
          xLabel="Date"
          yLabel="Resting HR (bpm)"
        />
      </div>
    </section>
  );
}
