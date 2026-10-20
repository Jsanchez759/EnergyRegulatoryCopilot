import { STATUS } from "./labels";

export default function StatusBadge({ status }) {
  const info = STATUS[status] || STATUS.desconocido;
  return <span className={`status-badge is-${status}`} title={info.hint}>{info.label}</span>;
}
