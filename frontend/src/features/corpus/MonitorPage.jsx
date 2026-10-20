import { Link, useSearchParams } from "react-router-dom";

import { useCorpusFeed, useCorpusStatus, useNorms, useSourceCatalog } from "./api";
import Deadline from "./Deadline";
import {
  eventLabel,
  FEED_FILTERS,
  formatDate,
  relativeDay,
  sourceName,
} from "./labels";
import StatusBadge from "./StatusBadge";

function Stats() {
  const status = useCorpusStatus();
  const totals = (status.data?.sources || []).reduce((sum, source) => {
    sum.indexed += source.indexed;
    Object.entries(source.by_status).forEach(([key, value]) => { sum[key] = (sum[key] || 0) + value; });
    return sum;
  }, { indexed: 0 });
  const tiles = [
    { label: "Normas indexadas", value: totals.indexed },
    { label: "Vigentes", value: totals.vigente || 0 },
    { label: "Proyectos", value: totals.proyecto || 0 },
    { label: "Derogadas", value: totals.derogada || 0 },
  ];
  return (
    <div className="corpus-stats" aria-label="Resumen del corpus">
      {tiles.map((tile) => <div key={tile.label} className="corpus-stat">
        <span>{status.isLoading ? "—" : tile.value.toLocaleString("es-CO")}</span>
        <small>{tile.label}</small>
      </div>)}
    </div>
  );
}

function FilterChips({ label, options, value, onChange }) {
  return (
    <div className="chip-group" role="group" aria-label={label}>
      {options.map((option) => <button
        key={option.id || "all"}
        type="button"
        className={`chip ${value === option.id ? "is-active" : ""}`}
        aria-pressed={value === option.id}
        onClick={() => onChange(option.id)}
      >{option.label}</button>)}
    </div>
  );
}

function FeedItem({ event, catalog }) {
  const { norm } = event;
  const statusChange = event.kind === "estado_cambiado" && event.details?.to;
  return (
    <li className="feed-item">
      <div className="feed-item-meta">
        <span className={`event-kind is-${event.kind}`}>{eventLabel(event)}</span>
        <span className="source-tag">{sourceName(catalog, norm.source)}</span>
        {norm.issued_on && <span>Expedida {formatDate(norm.issued_on)}</span>}
      </div>
      <Link className="feed-item-title" to={`/normativa/${norm.id}`}>{norm.label}</Link>
      {norm.title && <p className="feed-item-summary">{norm.title}</p>}
      <div className="feed-item-footer">
        <StatusBadge status={norm.status} />
        {statusChange && <span className="feed-item-note">Pasó a «{statusChange}»</span>}
        <Deadline norm={norm} />
      </div>
    </li>
  );
}

function DraftList({ items }) {
  return (
    <ul className="draft-list">
      {items.map((draft) => <li key={draft.id}>
        <Link to={`/normativa/${draft.id}`}>{draft.label}</Link>
        {draft.comment_end
          ? <Deadline norm={draft} />
          : <small>{draft.published_on ? `Publicado ${formatDate(draft.published_on)}` : formatDate(draft.issued_on)}</small>}
        {draft.title && <p>{draft.title}</p>}
      </li>)}
    </ul>
  );
}

function OpenConsultations({ source }) {
  const open = useNorms({ open_consultation: true, source, limit: 20 });
  const items = [...(open.data?.items || [])].sort((a, b) => a.comment_end.localeCompare(b.comment_end));
  if (!items.length) return null;
  return (
    <section className="monitor-panel" aria-label="Consultas abiertas">
      <h2>Consultas abiertas</h2>
      <p className="monitor-panel-note">Proyectos que todavía reciben comentarios, por fecha de cierre.</p>
      <DraftList items={items} />
      <Link className="monitor-panel-link" to="/normativa?open=1">Ver todas las consultas abiertas</Link>
    </section>
  );
}

function RecentDrafts({ source }) {
  const drafts = useNorms({ status: "proyecto", source, limit: 6 });
  const items = drafts.data?.items || [];
  return (
    <section className="monitor-panel" aria-label="Proyectos recientes">
      <h2>Proyectos recientes</h2>
      <p className="monitor-panel-note">Normas publicadas para comentarios. Si la fuente no publica el plazo, está en el texto del proyecto.</p>
      {drafts.isLoading ? <p className="monitor-empty">Cargando…</p> : items.length
        ? <DraftList items={items} />
        : <p className="monitor-empty">No hay proyectos indexados todavía.</p>}
      <Link className="monitor-panel-link" to="/normativa?status=proyecto">Ver todos los proyectos</Link>
    </section>
  );
}

export default function MonitorPage() {
  const [params, setParams] = useSearchParams();
  const source = params.get("source") || "";
  const kind = params.get("kind") || "";
  const catalog = useSourceCatalog();
  const activeFilter = FEED_FILTERS.find((filter) => filter.id === kind) || FEED_FILTERS[0];
  const feed = useCorpusFeed({ source, kind: activeFilter.kinds ? activeFilter.kinds.join(",") : kind });
  const events = feed.data?.pages.flatMap((page) => page.events) || [];
  const groups = events.reduce((result, event) => {
    const day = relativeDay(event.occurred_at);
    const group = result.at(-1);
    if (group?.day === day) group.events.push(event);
    else result.push({ day, events: [event] });
    return result;
  }, []);

  function update(key, value) {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    setParams(next, { replace: true });
  }

  const sourceOptions = [{ id: "", label: "Todas las fuentes" }, ...(catalog.data?.sources || []).map((item) => ({ id: item.id, label: item.name }))];

  return (
    <div className="monitor-page">
      <header className="corpus-hero">
        <div className="welcome-rule"><span /> MONITOR REGULATORIO</div>
        <h1>Qué cambió en la <em>regulación eléctrica.</em></h1>
        <p>Normas nuevas, textos modificados y cambios de vigencia detectados en las fuentes oficiales.</p>
      </header>
      <Stats />
      <div className="monitor-grid">
        <section className="monitor-feed" aria-label="Novedades">
          <div className="feed-filters">
            <FilterChips label="Fuente" options={sourceOptions} value={source} onChange={(value) => update("source", value)} />
            <FilterChips label="Tipo de cambio" options={FEED_FILTERS} value={activeFilter.id} onChange={(value) => update("kind", value)} />
          </div>
          {feed.isLoading && <p className="monitor-empty">Cargando novedades…</p>}
          {feed.isError && <p className="monitor-empty">No se pudieron cargar las novedades.</p>}
          {!feed.isLoading && !events.length && !feed.isError && <p className="monitor-empty">
            {activeFilter.id === "" ? "Sin novedades recientes. Las normas antiguas incorporadas al corpus aparecen en «Incorporadas»." : "No hay eventos con estos filtros."}
          </p>}
          {groups.map((group) => <div key={group.day} className="feed-day">
            <h2>{group.day}</h2>
            <ul>{group.events.map((event) => <FeedItem key={event.id} event={event} catalog={catalog.data} />)}</ul>
          </div>)}
          {feed.hasNextPage && <button className="load-more" type="button" onClick={() => feed.fetchNextPage()} disabled={feed.isFetchingNextPage}>
            {feed.isFetchingNextPage ? "Cargando…" : "Cargar más"}
          </button>}
        </section>
        <aside className="monitor-aside">
          <OpenConsultations source={source} />
          <RecentDrafts source={source} />
        </aside>
      </div>
    </div>
  );
}
