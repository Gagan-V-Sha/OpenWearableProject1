import { useEffect, useState } from "react";
import "./App.css";
import { getDashboard, getUsers } from "./api";
import { formatDateLong, greeting } from "./format";
import Header from "./components/Header";
import LandingHero from "./components/LandingHero";
import ScoreSection from "./components/ScoreSection";
import QuickOverview from "./components/QuickOverview";
import TrendsSection from "./components/TrendsSection";
import WhatIfPanel from "./components/WhatIfPanel";
import AskPanel from "./components/AskPanel";
import MonitoringPage from "./components/MonitoringPage";
import AuditPage from "./components/AuditPage";

function routeFromHash() {
  const h = window.location.hash;
  if (h.startsWith("#/monitoring")) return "monitoring";
  if (h.startsWith("#/audit")) return "audit";
  return "dashboard";
}

export default function App() {
  const [route, setRoute] = useState(routeFromHash);
  const [users, setUsers] = useState([]);
  const [userId, setUserId] = useState(null);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const onHash = () => {
      setRoute(routeFromHash());
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    getUsers()
      .then((list) => {
        setUsers(list);
        if (list.length > 0) setUserId(list[0].user_id);
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!userId) return;
    setData(null);
    getDashboard(userId)
      .then(setData)
      .catch((e) => setError(e.message));
  }, [userId]);

  const navigate = (r) => {
    window.location.hash = r === "dashboard" ? "#/" : `#/${r}`;
  };

  if (error) {
    return (
      <div className="page">
        <Header route={route} onNavigate={navigate} />
        <div className="card error-card">
          <h2>Can't reach the backend</h2>
          <p className="muted">{error}</p>
          <p>
            Start the API first: <code>cd Backend-Python/OpenWearableProject1</code> then{" "}
            <code>uvicorn api:app --reload --port 8000</code>
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      <Header route={route} onNavigate={navigate} />

      {route === "dashboard" && (
        <>
          <LandingHero data={data} />
          <div id="dashboard" className="dash">
            {!data ? (
              <div className="loading">Loading your recovery data…</div>
            ) : (
              <>
                <div className="greeting-row">
                  <div className="greeting">
                    <div className="greeting-date">{formatDateLong(data.window_end_date)}</div>
                    <h1>
                      {greeting()}, {data.display_name || data.user_id}
                    </h1>
                  </div>
                  <div className="dash-controls">
                    <span className="synced">
                      <span className="synced-dot" /> Synced
                    </span>
                    <select
                      className="user-select"
                      value={userId || ""}
                      onChange={(e) => setUserId(e.target.value)}
                    >
                      {users.map((u) => (
                        <option key={u.user_id} value={u.user_id}>
                          {u.display_name || u.user_id}
                          {u.age_group ? ` · ${u.age_group}` : ""}
                        </option>
                      ))}
                    </select>
                    <div className="avatar">{(userId || "A").slice(-1).toUpperCase()}</div>
                  </div>
                </div>

                <ScoreSection data={data} />
                <div id="features" className="anchor-wrap">
                  <QuickOverview data={data} />
                </div>
                <TrendsSection data={data} />
                <WhatIfPanel data={data} />
                <div id="use-cases" className="anchor-wrap">
                  <AskPanel data={data} />
                </div>
              </>
            )}
          </div>
        </>
      )}

      {route === "monitoring" && (
        <MonitoringPage
          users={users}
          userId={userId}
          onSelectUser={setUserId}
          data={data}
          onNavigate={navigate}
        />
      )}

      {route === "audit" && <AuditPage userId={userId} data={data} />}
    </div>
  );
}
