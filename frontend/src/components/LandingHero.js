import { formatSigned, LABEL_COLORS } from "../format";

function scrollTo(id) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

export default function LandingHero({ data }) {

  const score = data?.score ?? 42;
  const label = data?.label ?? "Poor";
  const rec = data?.recommendation ?? "Rest Day";
  const sleepPct = data?.metrics?.sleep?.change_pct;
  const hrBpm = data?.metrics?.resting_hr?.change_bpm;
  const evidence = [
    `sleep ${sleepPct != null ? formatSigned(sleepPct, 0, "%") : "-8%"} vs last week`,
    `resting HR ${hrBpm != null ? formatSigned(hrBpm, 1) : "+4"} bpm vs baseline`,
  ].join(" · ");
  const color = LABEL_COLORS[label] || "var(--accent)";

  return (
    <section className="hero">
      <div className="hero-copy">
        <span className="hero-badge">● Explainable recovery coach</span>
        <h1>
          Stop guessing.
          <br />
          <span className="hl-violet">Ask your body</span>{" "}
          <span className="hl-orange">a question.</span>
        </h1>
        <p>
          Your wearable already knows. Whyable reads your last 7 days, compares them to your
          30-day baseline, and answers in plain language — with the evidence behind every
          answer.
        </p>
        <button className="btn-cta" onClick={() => scrollTo("dashboard")}>
          Open dashboard ↓
        </button>
        <div className="hero-foot">
          Fitbit · Samsung Galaxy Watch · transparent rules — no black box
        </div>
      </div>

      <div className="terminal" id="how-it-works">
        <div className="terminal-bar">
          <span className="tdot tdot-red" />
          <span className="tdot tdot-amber" />
          <span className="tdot tdot-green" />
          <span className="terminal-title">whyable — ask</span>
        </div>
        <div className="terminal-body">
          <div className="term-line">
            › ask: <span className="term-q">Am I recovering well?</span>
          </div>
          <div className="term-answer">
            <div className="term-kicker">ANSWER</div>
            <div className="term-result" style={{ color }}>
              Recovery {score} · {label} → {rec}
            </div>
            <div className="term-evidence">{evidence}.</div>
          </div>
          <button className="term-try" onClick={() => scrollTo("use-cases")}>
            Try this yourself →
          </button>
        </div>
      </div>
    </section>
  );
}
