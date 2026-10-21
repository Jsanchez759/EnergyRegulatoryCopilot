import { Link } from "react-router-dom";

import { formatDateTime } from "../corpus/labels";
import { useAuditLog } from "./api";
import { actionLabel } from "./labels";

export default function AdminActivity() {
  const audit = useAuditLog();
  const entries = audit.data?.entries || [];
  return (
    <div className="admin-page">
      <header className="corpus-hero compact">
        <div className="welcome-rule"><span /> ADMINISTRACIÓN</div>
        <h1>Actividad <em>de los administradores.</em></h1>
        <p>Cada aprobación, suspensión, cambio de rol o eliminación queda registrada.</p>
      </header>
      {audit.isLoading && <p className="monitor-empty">Cargando actividad…</p>}
      {audit.isError && <p className="monitor-empty">No se pudo cargar la actividad.</p>}
      {audit.data && !entries.length && <p className="monitor-empty">Todavía no hay acciones registradas.</p>}
      {entries.length > 0 && <ol className="admin-audit">
        {entries.map((entry) => <li key={entry.id}>
          <time dateTime={entry.created_at}>{formatDateTime(entry.created_at)}</time>
          <p>
            <strong>{entry.actor_email}</strong> {actionLabel(entry.action)}{" "}
            {entry.action !== "delete" && entry.target_user_id
              ? <Link to={`/admin/usuarios?usuario=${entry.target_user_id}`}>{entry.target_email}</Link>
              : <strong>{entry.target_email}</strong>}
          </p>
        </li>)}
      </ol>}
    </div>
  );
}
