import { useEffect, useState } from "react";
import { postAsk } from "../api";

const SUGGESTIONS = [
  "Why is my recovery low today?",
  "What does a Poor score mean?",
  "Why a rest day?",
  "How is the score calculated?",
];

export default function AskPanel({ data }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setMessages([
      {
        role: "assistant",
        text: `Hi — your recovery is ${data.label} today. Want to know why, or what to do about it? Tap a question below.`,
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
        {SUGGESTIONS.map((q) => (
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
