const TABS = [
  { id: "dashboard", label: "My Dashboard" },
  { id: "monitoring", label: "Recovery Monitoring" },
  { id: "audit", label: "Fairness Audit" },
];

export default function Header({ route, onNavigate }) {
  return (
    <header className="topbar">
      <div className="brand">
        Whyable <span className="brand-dot">•</span> <span className="brand-ai">AI</span>
      </div>
      <nav className="tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={route === t.id ? "tab active" : "tab"}
            onClick={() => onNavigate(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>
      <div className="topbar-right">
        <button className="btn-ghost" disabled title="Coming soon">
          Sign in
        </button>
        <button className="btn-solid" title="Coming soon">
          Try free
        </button>
      </div>
    </header>
  );
}
