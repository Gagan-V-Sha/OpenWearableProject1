import { useRef, useState } from "react";
import { LABEL_COLORS } from "../format";

function fmtDate(iso) {
  if (!iso) return "";
  const d = new Date(`${iso}T00:00:00`);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

// Shared hover-tooltip state: charts render inside a relatively-positioned
// box and position the tip at the mouse cursor.
function useChartTip() {
  const boxRef = useRef(null);
  const [tip, setTip] = useState(null);
  const show = (e, content) => {
    const r = boxRef.current?.getBoundingClientRect();
    if (!r) return;
    setTip({ x: e.clientX - r.left, y: e.clientY - r.top, content });
  };
  const hide = () => setTip(null);
  return { boxRef, tip, show, hide };
}

function ChartTip({ tip }) {
  if (!tip) return null;
  return (
    <div className="chart-tip" style={{ left: tip.x, top: tip.y }}>
      {tip.content}
    </div>
  );
}

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

export function SparkBars({ values, titles, color = "var(--accent)", width = 170, height = 44 }) {
  const [hoverIdx, setHoverIdx] = useState(null);
  const vals = values || [];
  if (vals.length === 0) return <div style={{ height }} />;
  const present = vals.filter((v) => v != null);
  const max = present.length > 0 ? Math.max(...present, 1) : 1;
  const gap = 3;
  const bw = (width - gap * (vals.length - 1)) / vals.length;

  const barFill = (i, last, missing) => {
    if (missing) return "none";
    if (hoverIdx === i) return color;
    if (hoverIdx == null && last) return color;
    return "var(--bar)";
  };

  const barOpacity = (i, missing) => {
    if (missing) return hoverIdx === i ? 1 : 0.75;
    if (hoverIdx == null) return 1;
    return hoverIdx === i ? 1 : 0.35;
  };

  return (
    <svg
      width={width}
      height={height}
      className="spark"
      onMouseLeave={() => setHoverIdx(null)}
    >
      {vals.map((v, i) => {
        const x = i * (bw + gap);
        const last = i === vals.length - 1;
        const missing = v == null;
        if (missing) {
          return (
            <rect
              key={i}
              x={x}
              y={height - 4}
              width={bw}
              height={2}
              rx={1}
              fill="none"
              stroke={hoverIdx === i ? color : "var(--border)"}
              strokeWidth={hoverIdx === i ? 1.5 : 1}
              strokeDasharray="2 2"
              opacity={barOpacity(i, true)}
              onMouseEnter={() => setHoverIdx(i)}
            >
              {titles && titles[i] && <title>{titles[i]}</title>}
            </rect>
          );
        }
        const h = Math.max(2, (v / max) * (height - 2));
        return (
          <rect
            key={i}
            x={x}
            y={height - h}
            width={bw}
            height={h}
            rx={2}
            fill={barFill(i, last, false)}
            opacity={barOpacity(i, false)}
            onMouseEnter={() => setHoverIdx(i)}
          >
            {titles && titles[i] && <title>{titles[i]}</title>}
          </rect>
        );
      })}
    </svg>
  );
}

export function BarChart({ data, color = "var(--accent)", height = 90, format = (v) => v }) {
  // data: [{ date, value }]
  const { boxRef, tip, show, hide } = useChartTip();
  const [hoverIdx, setHoverIdx] = useState(null);
  const points = data || [];
  const width = 100; // percentage-based viewBox so it stretches
  if (points.length === 0) return <div style={{ height }} />;
  const present = points.map((d) => d.value).filter((v) => v != null);
  const max = present.length > 0 ? Math.max(...present, 1) : 1;
  const gap = points.length > 12 ? 1 : 2.5;
  const bw = (width - gap * (points.length - 1)) / points.length;

  const barFill = (i, last) => {
    if (hoverIdx === i) return color;
    if (hoverIdx == null && last) return color;
    return "var(--bar)";
  };

  const barOpacity = (i) => {
    if (hoverIdx == null) return 1;
    return hoverIdx === i ? 1 : 0.35;
  };

  const onBarEnter = (e, i, label) => {
    setHoverIdx(i);
    show(e, label);
  };

  const onBarLeave = () => {
    setHoverIdx(null);
    hide();
  };

  return (
    <div className="barchart-wrap">
      <div className="bar-yaxis" style={{ height }}>
        <span>{format(max)}</span>
        <span>0</span>
      </div>
      <div ref={boxRef} className="chart-box" style={{ flex: 1 }}>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="none"
          className="barchart"
          style={{ width: "100%", height }}
          onMouseLeave={onBarLeave}
        >
          {points.map((point, i) => {
            const x = i * (bw + gap);
            const last = i === points.length - 1;
            const missing = point.value == null;
            const label = `${fmtDate(point.date)} · ${
              missing ? "no data" : format(point.value)
            }`;
            if (missing) {
              return (
                <g key={i}>
                  <rect
                    x={x}
                    y={height - 5}
                    width={bw}
                    height={2}
                    rx={1}
                    fill="none"
                    stroke={hoverIdx === i ? color : "var(--border)"}
                    strokeWidth={hoverIdx === i ? 0.9 : 0.6}
                    strokeDasharray="2 2"
                    opacity={hoverIdx == null ? 0.8 : hoverIdx === i ? 1 : 0.35}
                  />
                  <rect
                    x={x}
                    y={0}
                    width={bw}
                    height={height}
                    fill="transparent"
                    onMouseMove={(e) => onBarEnter(e, i, label)}
                    onMouseLeave={onBarLeave}
                  />
                </g>
              );
            }
            const h = Math.max(2, (point.value / max) * (height - 4));
            return (
              <rect
                key={i}
                x={x}
                y={height - h}
                width={bw}
                height={h}
                rx={1.5}
                fill={barFill(i, last)}
                opacity={barOpacity(i)}
                onMouseMove={(e) => onBarEnter(e, i, label)}
                onMouseLeave={onBarLeave}
              />
            );
          })}
        </svg>
        <ChartTip tip={tip} />
      </div>
    </div>
  );
}

export function LineChart({ points, height = 180, unit = "", xLabel = "", yLabel = "" }) {
  // points: [{ date, value }]
  const { boxRef, tip, show, hide } = useChartTip();
  const [activeIdx, setActiveIdx] = useState(null);
  const data = (points || []).filter((p) => p.value != null);
  if (data.length < 2) {
    return <div className="chart-empty">Not enough data for this range.</div>;
  }
  const width = 800;
  const padL = yLabel ? 58 : 40;
  const padR = 16;
  const padY = 18;
  const padB = xLabel ? 48 : 34;
  const vals = data.map((p) => p.value);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const span = max - min || 1;
  const x = (i) => padL + (i / (data.length - 1)) * (width - padL - padR);
  const y = (v) => padY + (1 - (v - min) / span) * (height - padY - padB);
  const path = data.map((p, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(p.value)}`).join(" ");
  // Round every tick so raw floats (e.g. 78.2589…) don't overflow the axis.
  const fmtTick = (v) => (Math.abs(v) >= 10 ? Math.round(v) : Math.round(v * 10) / 10);
  const ticks = [fmtTick(max), fmtTick((max + min) / 2), fmtTick(min)];

  // Up to 4 evenly-spaced date labels along the x axis.
  const tickCount = Math.min(4, data.length);
  const xTicks = Array.from({ length: tickCount }, (_, k) =>
    Math.round((k / (tickCount - 1)) * (data.length - 1))
  );

  return (
    <div ref={boxRef} className="chart-box">
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
            <text x={padL - 6} y={y(t) + 4} className="axis-label" textAnchor="end">
              {t}
            </text>
          </g>
        ))}
        {yLabel && (
          <text
            transform={`rotate(-90 12 ${(height - padB) / 2 + padY / 2})`}
            x={12}
            y={(height - padB) / 2 + padY / 2}
            className="axis-label"
            textAnchor="middle"
          >
            {yLabel}
          </text>
        )}
        {xTicks.map((idx) => (
          <text
            key={idx}
            x={x(idx)}
            y={height - (xLabel ? 26 : 8)}
            className="axis-label"
            textAnchor={idx === 0 ? "start" : idx === data.length - 1 ? "end" : "middle"}
          >
            {fmtDate(data[idx].date)}
          </text>
        ))}
        {xLabel && (
          <text
            x={padL + (width - padL - padR) / 2}
            y={height - 6}
            className="axis-label"
            textAnchor="middle"
          >
            {xLabel}
          </text>
        )}
        <path d={path} fill="none" stroke="var(--accent)" strokeWidth={2} />
        {data.map((p, i) => (
          <g key={i}>
            <circle
              cx={x(i)}
              cy={y(p.value)}
              r={i === activeIdx ? 5.5 : 3.5}
              fill={i === activeIdx ? "var(--accent)" : "var(--bg)"}
              stroke="var(--text)"
              strokeWidth={1.5}
            />
            <circle
              cx={x(i)}
              cy={y(p.value)}
              r={12}
              fill="transparent"
              onMouseMove={(e) => {
                setActiveIdx(i);
                show(e, `${fmtDate(p.date)} · ${p.value}${unit ? ` ${unit}` : ""}`);
              }}
              onMouseLeave={() => {
                setActiveIdx(null);
                hide();
              }}
            />
          </g>
        ))}
      </svg>
      <ChartTip tip={tip} />
    </div>
  );
}
