// Set REACT_APP_API_BASE in frontend/.env (or the Vercel dashboard) to point
// at a hosted backend; defaults to the local dev server.
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

export function postAsk(body) {
  return request("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
