const number = new Intl.NumberFormat("es-CO", { maximumFractionDigits: 2 });
const small = new Intl.NumberFormat("es-CO", { maximumSignificantDigits: 3 });
const compact = new Intl.NumberFormat("es-CO", { maximumFractionDigits: 1 });

export function formatNumber(value) {
  // Below 0.01 two decimals would print every value as 0.
  return value !== 0 && Math.abs(value) < 0.01 ? small.format(value) : number.format(value);
}

export function formatCompact(value) {
  return compact.format(value);
}

export function isoDate(date) {
  const pad = (value) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

export function daysAgo(days, from = new Date()) {
  const date = new Date(from);
  date.setDate(date.getDate() - days);
  return isoDate(date);
}

// Fixed categorical order from the validated palette; "Otros" stays neutral.
export function seriesColor(name, index) {
  return name === "Otros" ? "var(--series-other)" : `var(--series-${(index % 6) + 1})`;
}
