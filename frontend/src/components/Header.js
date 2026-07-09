const LINKS = [
  { label: "Features", target: "features" },
  { label: "How it works", target: "how-it-works" },
];

function scrollTo(id) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

export default function Header() {
  return (
    <header className="topbar">
      <div className="brand">
        Whyable <span className="brand-dot">•</span> <span className="brand-ai">AI</span>
      </div>
      <nav className="nav-links">
        {LINKS.map((l) => (
          <button key={l.target} onClick={() => scrollTo(l.target)}>
            {l.label}
          </button>
        ))}
      </nav>
      <div className="topbar-right">
        <button className="btn-ghost" disabled title="Coming soon">
          Sign in
        </button>
        <button className="btn-solid" disabled title="Coming soon">
          Try free
        </button>
      </div>
    </header>
  );
}
