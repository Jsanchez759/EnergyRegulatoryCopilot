export const PROFILES = [
  { id: "comercializador", label: "Comercializador" },
  { id: "generador", label: "Generador" },
  { id: "red", label: "Transmisor u operador de red" },
  { id: "consultoria", label: "Consultoría o firma legal" },
  { id: "gobierno", label: "Entidad pública o regulador" },
  { id: "academia", label: "Academia o investigación" },
  { id: "otro", label: "Otro" },
];

export function profileLabel(id) {
  return PROFILES.find((profile) => profile.id === id)?.label || "";
}
