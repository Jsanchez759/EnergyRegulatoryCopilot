import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { profileLabel } from "../auth/profiles";
import { formatDate, formatDateTime } from "../corpus/labels";
import MarketChart from "../market/MarketChart";
import Icon from "../../shared/Icon";
import { useAdminAction, useAdminUser, useAdminUsers } from "./api";
import { ACCOUNT_STATUS, actionLabel, formatTokens, formatUsd } from "./labels";

const PAGE_SIZE = 25;
const FILTERS = [
  { id: "", label: "Todos", count: (counts) => counts.waitlist + counts.active + counts.suspended },
  { id: "waitlist", label: "En espera", count: (counts) => counts.waitlist },
  { id: "active", label: "Activos", count: (counts) => counts.active },
  { id: "suspended", label: "Suspendidos", count: (counts) => counts.suspended },
  { id: "admin", label: "Administradores", count: (counts) => counts.admin },
];

function StatusPill({ user }) {
  return (
    <span className="admin-pills">
      <span className={`admin-status is-${user.status}`}>{ACCOUNT_STATUS[user.status]}</span>
      {user.role === "admin" && <span className="admin-status is-admin">Admin</span>}
    </span>
  );
}

function Actions({ user, isSelf, onDone }) {
  const action = useAdminAction();
  const [confirming, setConfirming] = useState(false);
  if (isSelf) return <p className="admin-note">Es tu cuenta: otro administrador debe cambiar su estado o su rol.</p>;

  function run(kind, changes) {
    action.mutate({ userId: user.id, action: kind, changes }, { onSuccess: () => { setConfirming(false); if (kind === "delete") onDone(); } });
  }

  const waiting = user.status === "waitlist";
  return (
    <div className="admin-actions">
      {waiting && <button type="button" className="primary-action" disabled={action.isPending} onClick={() => run("update", { status: "active" })}><Icon name="check" size={15} /> Aprobar acceso</button>}
      {user.status === "suspended" && <button type="button" className="primary-action" disabled={action.isPending} onClick={() => run("update", { status: "active" })}>Reactivar</button>}
      {user.status === "active" && <>
        <button type="button" className="market-card-button" disabled={action.isPending} onClick={() => run("update", { role: user.role === "admin" ? "user" : "admin" })}>{user.role === "admin" ? "Quitar administrador" : "Hacer administrador"}</button>
        <button type="button" className="market-card-button" disabled={action.isPending} onClick={() => run("revoke")}>Cerrar sesiones</button>
        <button type="button" className="market-card-button is-warning" disabled={action.isPending} onClick={() => run("update", { status: "suspended" })}>Suspender</button>
      </>}
      {confirming
        ? <span className="admin-confirm">
          <span>{waiting ? "¿Rechazar y borrar la solicitud?" : "Se borran sus conversaciones y documentos."}</span>
          <button type="button" className="market-card-button is-danger" disabled={action.isPending} onClick={() => run("delete")}>Confirmar</button>
          <button type="button" className="market-card-button is-quiet" onClick={() => setConfirming(false)}>Cancelar</button>
        </span>
        : <button type="button" className="market-card-button is-danger" onClick={() => setConfirming(true)}>{waiting ? "Rechazar" : "Eliminar cuenta"}</button>}
      {action.isError && <p className="form-error" role="alert">{action.error.message}</p>}
    </div>
  );
}

function UserDetail({ userId, currentUserId, onClose }) {
  const detail = useAdminUser(userId);
  const user = detail.data;
  useEffect(() => {
    const onKey = (event) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <aside className="admin-drawer" aria-label="Detalle del usuario">
      <header className="admin-drawer-head">
        <div>
          <span className="sidebar-label">Usuario</span>
          <h2>{user?.display_name || "Cargando…"}</h2>
          {user && <small>{user.email}</small>}
        </div>
        <button type="button" className="market-card-button is-quiet" aria-label="Cerrar detalle" onClick={onClose}>×</button>
      </header>
      {detail.isError && <p className="monitor-empty">{detail.error.message}</p>}
      {user && <div className="admin-drawer-body">
        <StatusPill user={user} />
        <Actions user={user} isSelf={user.id === currentUserId} onDone={onClose} />
        <dl className="admin-facts">
          <div><dt>Empresa</dt><dd>{user.company || "—"}</dd></div>
          <div><dt>Perfil</dt><dd>{profileLabel(user.profile) || "—"}</dd></div>
          <div><dt>Registro</dt><dd>{formatDate(user.created_at)}</dd></div>
          <div><dt>Aprobado</dt><dd>{user.approved_at ? formatDate(user.approved_at) : "—"}</dd></div>
          <div><dt>Último acceso</dt><dd>{user.last_login_at ? formatDateTime(user.last_login_at) : "Nunca"}</dd></div>
          <div><dt>Sesiones activas</dt><dd>{user.active_sessions}</dd></div>
          <div><dt>Conversaciones</dt><dd>{user.conversations}</dd></div>
          <div><dt>Documentos</dt><dd>{user.documents}</dd></div>
        </dl>
        {user.use_case && <blockquote className="admin-use-case"><span className="sidebar-label">Para qué lo usaría</span>{user.use_case}</blockquote>}
        <section>
          <h3>Uso de los asistentes</h3>
          <dl className="admin-facts is-usage">
            <div><dt>Consultas · 30 días</dt><dd>{user.usage_30d.runs}{user.usage_30d.failed > 0 && <small> · {user.usage_30d.failed} fallidas</small>}</dd></div>
            <div><dt>Costo · 30 días</dt><dd>{formatUsd(user.usage_30d.cost_usd)}</dd></div>
            <div><dt>Tokens · 30 días</dt><dd>{formatTokens(user.usage_30d.prompt_tokens + user.usage_30d.completion_tokens)}</dd></div>
            <div><dt>Costo total</dt><dd>{formatUsd(user.usage_all_time.cost_usd)} <small>en {user.usage_all_time.runs} consultas</small></dd></div>
          </dl>
          {user.usage_30d.runs > 0 && <MarketChart
            source={{ id: "", title: "Consultas por día", chart: { x: user.daily.map((day) => day.day), series: [{ name: "Consultas", values: user.daily.map((day) => day.runs) }], unit: "consultas" } }}
            type="bar"
            height={150}
            caption={false}
          />}
        </section>
        {user.audit.length > 0 && <section>
          <h3>Historial</h3>
          <ol className="admin-audit is-compact">
            {user.audit.map((entry) => <li key={entry.id}>
              <time dateTime={entry.created_at}>{formatDateTime(entry.created_at)}</time>
              <p><strong>{entry.actor_email}</strong> {actionLabel(entry.action)} esta cuenta</p>
            </li>)}
          </ol>
        </section>}
      </div>}
    </aside>
  );
}

function QuickApprove({ user }) {
  const action = useAdminAction();
  return <button
    type="button"
    className="market-card-button is-accent"
    disabled={action.isPending}
    onClick={(event) => { event.stopPropagation(); action.mutate({ userId: user.id, action: "update", changes: { status: "active" } }); }}
  >{action.isPending ? "Aprobando…" : "Aprobar"}</button>;
}

export default function AdminUsers({ currentUserId }) {
  const [params, setParams] = useSearchParams();
  const filter = params.get("estado") || "";
  const selected = params.get("usuario");
  const page = Number(params.get("pagina") || 0);
  const [text, setText] = useState(params.get("q") || "");
  const query = params.get("q") || "";
  const users = useAdminUsers({
    q: query,
    status: filter === "admin" ? "" : filter,
    role: filter === "admin" ? "admin" : "",
    limit: PAGE_SIZE,
    offset: page * PAGE_SIZE,
  });

  function update(changes) {
    const next = new URLSearchParams(params);
    Object.entries(changes).forEach(([key, value]) => {
      if (value || value === 0) next.set(key, String(value));
      else next.delete(key);
    });
    setParams(next, { replace: true });
  }

  useEffect(() => {
    const term = text.trim();
    if (term === query) return undefined;
    const timeout = window.setTimeout(() => {
      const next = new URLSearchParams(params);
      if (term) next.set("q", term);
      else next.delete("q");
      next.delete("pagina");
      setParams(next, { replace: true });
    }, 300);
    return () => window.clearTimeout(timeout);
  }, [text, query, params, setParams]);

  const counts = users.data?.counts;
  const total = users.data?.total || 0;
  const pages = Math.ceil(total / PAGE_SIZE);
  const closeDetail = () => update({ usuario: "" });

  return (
    <div className={`admin-page ${selected ? "has-drawer" : ""}`}>
      <header className="corpus-hero compact">
        <div className="welcome-rule"><span /> ADMINISTRACIÓN</div>
        <h1>Usuarios <em>y solicitudes.</em></h1>
        <p>Aprueba a quienes esperan en la lista, revisa su uso y gestiona el acceso.</p>
      </header>
      <div className="norms-filters admin-filters">
        <div className="norms-search-box admin-search">
          <Icon name="search" size={16} />
          <input type="search" value={text} onChange={(event) => setText(event.target.value)} placeholder="Busca por nombre, correo o empresa" aria-label="Buscar usuarios" />
        </div>
        <div className="chip-group" role="group" aria-label="Estado">
          {FILTERS.map((option) => <button
            key={option.id || "all"}
            type="button"
            className={`chip ${filter === option.id ? "is-active" : ""} ${option.id === "waitlist" && counts?.waitlist ? "has-pending" : ""}`}
            aria-pressed={filter === option.id}
            onClick={() => update({ estado: option.id, pagina: "" })}
          >{option.label}{counts && <span className="chip-count">{option.count(counts)}</span>}</button>)}
        </div>
      </div>
      {users.isLoading && <p className="monitor-empty">Cargando usuarios…</p>}
      {users.isError && <p className="monitor-empty">No se pudieron cargar los usuarios.</p>}
      {users.data && (users.data.users.length ? <div className="admin-table-wrap">
        <table className="admin-table">
          <thead><tr><th>Usuario</th><th>Empresa</th><th>Estado</th><th>Registro</th><th>Último acceso</th><th>Consultas · 30 d</th><th>Costo · 30 d</th><th aria-label="Acciones" /></tr></thead>
          <tbody>{users.data.users.map((user) => <tr key={user.id} className={user.id === selected ? "is-selected" : ""} onClick={() => update({ usuario: user.id })}>
            <td><button type="button" className="admin-row-link" onClick={(event) => { event.stopPropagation(); update({ usuario: user.id }); }}>{user.display_name}</button><small>{user.email}</small></td>
            <td>{user.company || "—"}<small>{profileLabel(user.profile)}</small></td>
            <td><StatusPill user={user} /></td>
            <td className="is-date">{formatDate(user.created_at)}</td>
            <td className="is-date">{user.last_login_at ? formatDate(user.last_login_at) : "Nunca"}</td>
            <td>{user.runs_30d.toLocaleString("es-CO")}</td>
            <td>{formatUsd(user.cost_30d)}</td>
            <td>{user.status === "waitlist" && <QuickApprove user={user} />}</td>
          </tr>)}</tbody>
        </table>
      </div> : <p className="monitor-empty">Ningún usuario coincide con estos filtros.</p>)}
      {pages > 1 && <nav className="pagination" aria-label="Paginación">
        <button type="button" onClick={() => update({ pagina: page - 1 || "" })} disabled={page === 0}>Anterior</button>
        <span>Página {page + 1} de {pages}</span>
        <button type="button" onClick={() => update({ pagina: page + 1 })} disabled={page + 1 >= pages}>Siguiente</button>
      </nav>}
      {selected && <UserDetail key={selected} userId={selected} currentUserId={currentUserId} onClose={closeDetail} />}
    </div>
  );
}
