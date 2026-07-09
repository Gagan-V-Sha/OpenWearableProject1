export function formatHours(hours) {
  if (hours == null) return "–";
  const h = Math.floor(hours);
  const m = Math.round((hours - h) * 60);
  return `${h}h ${String(m).padStart(2, "0")}m`;
}

export function formatSteps(steps) {
  if (steps == null) return "–";
  if (steps >= 1000) return `${(steps / 1000).toFixed(1)}k`;
  return String(Math.round(steps));
}

export function formatSigned(value, digits = 0, suffix = "") {
  if (value == null) return "–";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}${suffix}`;
}

export function formatDateLong(iso) {
  if (!iso) return "";
  const d = new Date(`${iso}T00:00:00`);
  return d
    .toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric" })
    .toUpperCase()
    .replace(",", " ·");
}

export function greeting() {
  const h = new Date().getHours();
  if (h < 12) return "Good morning";
  if (h < 18) return "Good afternoon";
  return "Good evening";
}

export const LABEL_COLORS = {
  Poor: "var(--orange)",
  Moderate: "var(--yellow)",
  Good: "var(--green)",
};
