export const STATUS = {
  vigente: { label: "Vigente", hint: "Vigente según la fuente" },
  derogada: { label: "Derogada", hint: "Derogada según la fuente" },
  proyecto: { label: "Proyecto", hint: "Proyecto publicado para comentarios" },
  desconocido: { label: "Sin verificar", hint: "Aún no se ha procesado su texto" },
};

export const EVENT_KINDS = {
  norma_nueva: "Nueva",
  norma_actualizada: "Texto actualizado",
  estado_cambiado: "Cambio de estado",
  norma_incorporada: "Incorporada al corpus",
};

export const FEED_FILTERS = [
  { id: "", label: "Novedades", kinds: ["norma_nueva", "norma_actualizada", "estado_cambiado"] },
  { id: "norma_nueva", label: "Nuevas" },
  { id: "norma_actualizada", label: "Actualizadas" },
  { id: "estado_cambiado", label: "Cambios de estado" },
  { id: "norma_incorporada", label: "Incorporadas" },
];

const RELATION_PAST = {
  modifica: "modificó",
  adiciona: "adicionó",
  deroga: "derogó",
  sustituye: "sustituyó",
  suspende: "suspendió",
  corrige: "corrigió",
  reglamenta: "reglamentó",
  prorroga: "prorrogó",
};

export function relationVerb(relation) {
  return RELATION_PAST[relation] || relation;
}

export function sourceName(catalog, sourceId) {
  return catalog?.sources?.find((source) => source.id === sourceId)?.name || sourceId.toUpperCase();
}

export function docTypeLabel(catalog, sourceId, docType) {
  const source = catalog?.sources?.find((item) => item.id === sourceId);
  return source?.doc_types?.[docType] || docType.replace(/_/g, " ");
}

export function allDocTypes(catalog, sourceId) {
  const sources = (catalog?.sources || []).filter((source) => !sourceId || source.id === sourceId);
  const types = new Map();
  sources.forEach((source) => Object.entries(source.doc_types).forEach(([id, label]) => types.set(id, label)));
  return [...types.entries()].map(([id, label]) => ({ id, label }));
}

const dateFormat = new Intl.DateTimeFormat("es-CO", { day: "numeric", month: "short", year: "numeric" });
const timeFormat = new Intl.DateTimeFormat("es-CO", { hour: "2-digit", minute: "2-digit" });

export function formatDate(value) {
  if (!value) return "";
  // Plain dates are calendar days; parsing them as UTC would shift them a day back in Colombia.
  const date = /^\d{4}-\d{2}-\d{2}$/.test(value) ? new Date(`${value}T12:00:00`) : new Date(value);
  return dateFormat.format(date);
}

export function formatDateTime(value) {
  if (!value) return "";
  const date = new Date(value);
  return `${dateFormat.format(date)} · ${timeFormat.format(date)}`;
}

export function relativeDay(value) {
  const date = new Date(value);
  const today = new Date();
  const days = Math.round((startOfDay(today) - startOfDay(date)) / 86_400_000);
  if (days === 0) return "Hoy";
  if (days === 1) return "Ayer";
  return dateFormat.format(date);
}

function startOfDay(date) {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
}
