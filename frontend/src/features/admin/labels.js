export const ACCOUNT_STATUS = {
  waitlist: "En espera",
  active: "Activo",
  suspended: "Suspendido",
};

export const FEATURES = { research: "Investigación", market: "Asistente de mercado" };

const ACTIONS = {
  approve: "aprobó a",
  reactivate: "reactivó a",
  suspend: "suspendió a",
  move_to_waitlist: "devolvió a la lista de espera a",
  promote: "hizo administrador a",
  demote: "quitó el rol de administrador a",
  revoke_sessions: "cerró las sesiones de",
  delete: "eliminó la cuenta de",
};

export function actionLabel(action) {
  return ACTIONS[action] || action;
}

const usd = new Intl.NumberFormat("es-CO", { style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 2 });
const smallUsd = new Intl.NumberFormat("es-CO", { style: "currency", currency: "USD", minimumFractionDigits: 4, maximumFractionDigits: 4 });

export function formatUsd(value) {
  if (!value) return usd.format(0);
  return Math.abs(value) < 0.1 ? smallUsd.format(value) : usd.format(value);
}

export function formatTokens(value) {
  return new Intl.NumberFormat("es-CO", { notation: value >= 10_000 ? "compact" : "standard", maximumFractionDigits: 1 }).format(value || 0);
}
