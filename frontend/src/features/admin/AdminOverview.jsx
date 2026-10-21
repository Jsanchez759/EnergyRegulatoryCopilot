import { Link } from "react-router-dom";

import MarketChart from "../market/MarketChart";
import { useAdminOverview } from "./api";
import { FEATURES, formatTokens, formatUsd } from "./labels";

function dailyChart(daily, field, name, unit) {
  return {
    id: "",
    title: name,
    chart: { x: daily.map((day) => day.day), series: [{ name, values: daily.map((day) => day[field]) }], unit },
  };
}

function Breakdown({ title, rows, label = (name) => name }) {
  return (
    <section className="admin-panel">
      <h2>{title}</h2>
      {rows.length ? <table className="admin-table is-compact">
        <thead><tr><th>Nombre</th><th>Consultas</th><th>Costo</th></tr></thead>
        <tbody>{rows.map((row) => <tr key={row.name}><td>{label(row.name)}</td><td>{row.runs.toLocaleString("es-CO")}</td><td>{formatUsd(row.cost_usd)}</td></tr>)}</tbody>
      </table> : <p className="monitor-empty">Sin consultas en los últimos 30 días.</p>}
    </section>
  );
}

export default function AdminOverview() {
  const overview = useAdminOverview();
  if (overview.isLoading) return <p className="monitor-empty">Cargando resumen…</p>;
  if (overview.isError) return <p className="monitor-empty">No se pudo cargar el resumen.</p>;
  const { users, usage_7d: week, usage_30d: month, daily } = overview.data;
  const perRun = month.runs ? month.cost_usd / month.runs : 0;
  const tiles = [
    { label: "Usuarios activos", value: users.active.toLocaleString("es-CO"), detail: `${users.using_7d} ${users.using_7d === 1 ? "usó" : "usaron"} los asistentes esta semana` },
    { label: "En lista de espera", value: users.waitlist.toLocaleString("es-CO"), detail: users.waitlist ? <Link to="/admin/usuarios?estado=waitlist">Revisar solicitudes</Link> : "Nada pendiente", highlight: users.waitlist > 0 },
    { label: "Consultas · 7 días", value: week.runs.toLocaleString("es-CO"), detail: `${month.runs.toLocaleString("es-CO")} en 30 días · ${month.failed} fallidas` },
    { label: "Costo de modelos · 30 días", value: formatUsd(month.cost_usd), detail: `${formatUsd(perRun)} por consulta · ${formatTokens(month.prompt_tokens + month.completion_tokens)} tokens` },
  ];

  return (
    <div className="admin-page">
      <header className="corpus-hero compact">
        <div className="welcome-rule"><span /> ADMINISTRACIÓN</div>
        <h1>Cómo va <em>la beta.</em></h1>
        <p>Usuarios, uso de los asistentes y lo que cuesta cada consulta a los modelos.</p>
      </header>
      <div className="corpus-stats admin-stats" aria-label="Resumen">
        {tiles.map((tile) => <div key={tile.label} className={`corpus-stat ${tile.highlight ? "is-highlight" : ""}`}>
          <span>{tile.value}</span>
          <small>{tile.label}</small>
          <em>{tile.detail}</em>
        </div>)}
      </div>
      <div className="admin-charts">
        <section className="admin-panel">
          <h2>Consultas por día</h2>
          <MarketChart source={dailyChart(daily, "runs", "Consultas", "consultas")} type="bar" height={200} caption={false} />
        </section>
        <section className="admin-panel">
          <h2>Costo por día</h2>
          <MarketChart source={dailyChart(daily, "cost_usd", "Costo", "USD")} type="bar" height={200} caption={false} />
        </section>
      </div>
      <div className="admin-charts">
        <Breakdown title="Por herramienta · 30 días" rows={overview.data.by_feature} label={(name) => FEATURES[name] || name} />
        <Breakdown title="Por modelo · 30 días" rows={overview.data.by_model} />
      </div>
      <section className="admin-panel">
        <h2>Usuarios con más uso · 30 días</h2>
        {overview.data.top_users.length ? <table className="admin-table is-compact">
          <thead><tr><th>Usuario</th><th>Consultas</th><th>Costo</th></tr></thead>
          <tbody>{overview.data.top_users.map((user) => <tr key={user.id || user.email}>
            <td>{user.id ? <Link to={`/admin/usuarios?usuario=${user.id}`}>{user.display_name}</Link> : user.display_name}<small>{user.email}</small></td>
            <td>{user.runs.toLocaleString("es-CO")}</td>
            <td>{formatUsd(user.cost_usd)}</td>
          </tr>)}</tbody>
        </table> : <p className="monitor-empty">Todavía no hay consultas registradas.</p>}
      </section>
    </div>
  );
}
