import { useEffect, useRef, useState } from "react";
import { postWhatIf } from "../api";
import { Gauge } from "./Charts";
import { formatHours, formatSigned } from "../format";

function loadLevel(ratio) {
  if (ratio > 1.2) return "High";
  if (ratio < 0.8) return "Low";
  return "Balanced";
}

export default function WhatIfPanel({ data }) {
  const baseSleep = data.current.sleep_avg_7d ?? 7;
  const baseLoad = data.current.training_load_ratio ?? 1;
  const [sleep, setSleep] = useState(baseSleep);
  const [load, setLoad] = useState(baseLoad);
  const [result, setResult] = useState(null);
  const timer = useRef(null);

  useEffect(() => {
    setSleep(data.current.sleep_avg_7d ?? 7);
    setLoad(data.current.training_load_ratio ?? 1);
    setResult(null);
  }, [data]);

  useEffect(() => {
    clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      postWhatIf({
        user_id: data.user_id,
        sleep_hours: sleep,
        training_load_ratio: load,
      })
        .then(setResult)
        .catch(() => setResult(null));
    }, 250);
    return () => clearTimeout(timer.current);
  }, [data.user_id, sleep, load]);

  return (
    <section className="card whatif-card">
      <div className="whatif-main">
        <h3>✦ What if you changed something?</h3>
        <p className="muted">
          Drag the sliders to see how tonight's choices could move tomorrow's recovery.
        </p>

        <div className="slider-row">
          <label>Tonight's sleep</label>
          <span className="slider-value">{formatHours(sleep)}</span>
        </div>
        <input
          type="range"
          min="3"
          max="10"
          step="0.25"
          value={sleep}
          onChange={(e) => setSleep(Number(e.target.value))}
        />

        <div className="slider-row">
          <label>Tomorrow's training load</label>
          <span className="slider-value">
            {loadLevel(load)} · {load.toFixed(2)}x
          </span>
        </div>
        <input
          type="range"
          min="0"
          max="2"
          step="0.05"
          value={load}
          onChange={(e) => setLoad(Number(e.target.value))}
          className="range-orange"
        />

        {result && (
          <div className="whatif-message">
            <span className="whatif-bulb">💡</span> {result.message}
          </div>
        )}
      </div>

      <div className="whatif-result">
        <div className="card-kicker">PREDICTED RECOVERY</div>
        <Gauge
          score={result ? result.predicted_score : data.score}
          label={result ? result.predicted_label : data.label}
          caption={result ? result.predicted_label : data.label}
          size={130}
        />
        {result && (
          <div className="muted small">
            {formatSigned(result.delta)} vs today ({result.current_score})
          </div>
        )}
      </div>
    </section>
  );
}
