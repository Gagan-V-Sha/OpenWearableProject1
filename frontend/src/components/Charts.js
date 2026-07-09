import { LABEL_COLORS } from "../format";

export function Gauge({ score, label, caption, size = 150, stroke = 10 }) {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, score ?? 0));
  const color = LABEL_COLORS[label] || "var(--accent)";
  return (
    <div className="gauge" style={{ width: size, height: size }}>
      <svg width={size} height={size}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="var(--border)"
          strokeWidth={stroke}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${(pct / 100) * c} ${c}`}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </svg>
      <div className="gauge-center">
        <div className="gauge-score">{score ?? "–"}</div>
        <div className="gauge-caption">{caption}</div>
      </div>
    </div>
  );
}

export function SparkBars({ values, color = "var(--accent)", width = 170, height = 44 }) {
  const vals = (values || []).map((v) => (v == null ? 0 : v));
  if (vals.length === 0) return <div style={{ height }} />;
  const max = Math.max(...vals, 1);
  const gap = 3;
  const bw = (width - gap * (vals.length - 1)) / vals.length;
  return (
    <svg width={width} height={height} className="spark">
      {vals.map((v, i) => {
        const h = Math.max(2, (v / max) * (height - 2));
        const last = i === vals.length - 1;
        return (
          <rect
            key={i}
            x={i * (bw + gap)}
            y={height - h}
            width={bw}
            height={h}
            rx={2}
            fill={last ? color : "var(--bar)"}
          />
        );
      })}
    </svg>
  );
}

export function BarChart({ values, color = "var(--accent)", height = 90 }) {
  const vals = (values || []).map((v) => (v == null ? 0 : v));
  const width = 100;
  if (vals.length === 0) return <div style={{ height }} />;
  const max = Math.max(...vals, 1);
  const gap = vals.length > 12 ? 1 : 2.5;
  const bw = (width - gap * (vals.length - 1)) / vals.length;
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className="barchart"
      style={{ width: "100%", height }}
    >
      {vals.map((v, i) => {
        const h = Math.max(2, (v / max) * (height - 4));
        const last = i === vals.length - 1;
        return (
          <rect
            key={i}
            x={i * (bw + gap)}
            y={height - h}
            width={bw}
            height={h}
            rx={1.5}
            fill={last ? color : "var(--bar)"}
          />
        );
      })}
    </svg>
  );
}

export function LineChart({ points, height = 180 }) {

  const data = (points || []).filter((p) => p.value != null);
  if (data.length < 2) {
    return <div className="chart-empty">Not enough heart-rate data for this range.</div>;
  }
  const width = 800;
  const padL = 34;
  const padR = 16;
  const padY = 18;
  const vals = data.map((p) => p.value);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const span = max - min || 1;
  const x = (i) => padL + (i / (data.length - 1)) * (width - padL - padR);
  const y = (v) => padY + (1 - (v - min) / span) * (height - padY * 2);
  const path = data.map((p, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(p.value)}`).join(" ");
  const ticks = [max, Math.round((max + min) / 2), min];
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="linechart" style={{ width: "100%" }}>
      {ticks.map((t, i) => (
        <g key={i}>
          <line
            x1={padL}
            x2={width - padR}
            y1={y(t)}
            y2={y(t)}
            stroke="var(--border)"
            strokeDasharray="3 5"
          />
          <text x={4} y={y(t) + 4} className="axis-label">
            {t}
          </text>
        </g>
      ))}
      <path d={path} fill="none" stroke="var(--accent)" strokeWidth={2} />
      {data.map((p, i) => (
        <circle
          key={i}
          cx={x(i)}
          cy={y(p.value)}
          r={3.5}
          fill="var(--bg)"
          stroke="var(--text)"
          strokeWidth={1.5}
        />
      ))}
      <text x={padL} y={height - 2} className="axis-label">
        {data[0].date.slice(5)}
      </text>
      <text x={width - padR} y={height - 2} className="axis-label" textAnchor="end">
        {data[data.length - 1].date.slice(5)}
      </text>
    </svg>
  );
}
