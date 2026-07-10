import { useEffect, useMemo, useState } from "react";
import { postAsk } from "../api";

const CALCULATED = "How is the score calculated?";

const SUGGESTIONS_BY_LABEL = {
  Poor: [
    "Why is my recovery low today?",
    "What does a Poor score mean?",
    "Why a rest day?",
    CALCULATED,
  ],
  Moderate: [
    "Why light activity today?",
    "What should I change to improve?",
    "Am I recovering well?",
    CALCULATED,
  ],
  Good: [
    "Why can I train hard today?",
    "What does a Good score mean?",
    "Am I recovering well?",
    CALCULATED,
  ],
};

function openingMessage(data) {
  const label = data?.label ?? "Poor";
  const rec = data?.recommendation ?? "Rest Day";
  if (label === "Good") {
    return `Hi — your recovery is ${label} today (${rec}). Want to know what's driving it? Tap a question below.`;
  }
  if (label === "Moderate") {
    return `Hi — your recovery is ${label} today (${rec}). Want to know why, or how to push it higher? Tap a question below.`;
  }
  return `Hi — your recovery is ${label} today. Want to know why, or what to do about it? Tap a question below.`;
}

export default function AskPanel({ data }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  const suggestions = useMemo(
    () => SUGGESTIONS_BY_LABEL[data?.label] ?? SUGGESTIONS_BY_LABEL.Poor,
    [data?.label]
  );

  useEffect(() => {
    setMessages([
      {
        role: "assistant",
        text: openingMessage(data),
      },
    ]);
  }, [data]);

  async function ask(question) {
    if (!question.trim() || busy) return;
    setInput("");
    setBusy(true);
    const prior = messages;
    setMessages((m) => [...m, { role: "user", text: question }]);
    try {
      const res = await postAsk({
        user_id: data.user_id,
        question,
        history: prior.map((m) => ({ role: m.role, text: m.text })),
      });
      setMessages((m) => [...m, { role: "assistant", text: res.answer }]);
    } catch {
      setMessages((m) => [
        ...m,
        { role: "assistant", text: "Sorry, I couldn't reach the backend. Is the API running?" },
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="card ask-card">
      <h3>
        <span className="ask-dot">●</span> Ask about your scores
      </h3>
      <p className="muted">
        I only answer questions about your recovery score, sleep, training, and recommendations.
      </p>

      <div className="ask-thread">
        {messages.map((m, i) => (
          <div key={i} className={`bubble bubble-${m.role}`}>
            {m.text}
          </div>
        ))}
        {busy && <div className="bubble bubble-assistant muted">Thinking…</div>}
      </div>

      <div className="ask-chips">
        {suggestions.map((q) => (
          <button key={q} className="chip" onClick={() => ask(q)} disabled={busy}>
            {q}
          </button>
        ))}
      </div>

      <form
        className="ask-input"
        onSubmit={(e) => {
          e.preventDefault();
          ask(input);
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about your score, sleep, or training…"
        />
        <button type="submit" disabled={busy || !input.trim()}>
          Ask
        </button>
      </form>
    </section>
  );
}
