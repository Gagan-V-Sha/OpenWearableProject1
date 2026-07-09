import { useState } from "react";
import { BarChart, LineChart } from "./Charts";
import { formatHours, formatSteps } from "../format";

const RANGES = [3, 7, 14, 30];

function avg(values) {
  const v = values.filter((x) => x != null);
  if (v.length === 0) return null;
  return v.reduce((a, b) => a + b, 0) / v.length;
}

function TrendCard({ title, avgLabel, values, color, range }) {
  return (
    <div className="card trend-card">
      <div className="trend-head">
        <span>{title}</span>
        <span className="muted">avg {avgLabel}</span>
      </div>
      <BarChart values={values} color={color} />
      <div className="trend-axis">
        <span>-{range}d</span>
        <span>today</span>
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
          values={slice.map((d) => d.sleep_hours)}
          color="var(--orange)"
          range={range}
        />
        <TrendCard
          title="Activity"
          avgLabel={formatSteps(avg(slice.map((d) => d.steps)))}
          values={slice.map((d) => d.steps)}
          color="var(--blue)"
          range={range}
        />
        <TrendCard
          title="Training load"
          avgLabel={`${Math.round(avg(slice.map((d) => d.active_minutes)) ?? 0)} min`}
          values={slice.map((d) => d.active_minutes)}
          color="var(--green)"
          range={range}
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
        <LineChart points={hrSlice.map((d) => ({ date: d.date, value: d.resting_hr }))} />
      </div>
    </section>
  );
}
