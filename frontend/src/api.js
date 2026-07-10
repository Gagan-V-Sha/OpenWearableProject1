const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";

async function request(path, options) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

export function getUsers() {
  return request("/api/users");
}

export function getDashboard(userId) {
  return request(`/api/dashboard/${encodeURIComponent(userId)}?days=30`);
}

export function postWhatIf(body) {
  return request("/api/whatif", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function getMonitoring(userId, days = 120) {
  return request(`/api/monitoring/${encodeURIComponent(userId)}?days=${days}`);
}

export function getAudit(userId) {
  return request(`/api/audit/${encodeURIComponent(userId)}`);
}

export function getSuggest(userId) {
  return request(`/api/suggestions/${encodeURIComponent(userId)}`);
}

export function postAsk(body) {
  return request("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
