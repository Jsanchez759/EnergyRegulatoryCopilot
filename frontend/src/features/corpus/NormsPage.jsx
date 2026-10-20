import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { useCorpusSearch, useNorms, useSourceCatalog } from "./api";
import { allDocTypes, formatDate, sourceName, STATUS } from "./labels";
import StatusBadge from "./StatusBadge";
import Icon from "../../shared/Icon";

const PAGE_SIZE = 20;
const FIRST_YEAR = 2015;

function excerpt(text, query, size = 320) {
  const words = query.toLowerCase().split(/\s+/).filter((word) => word.length > 3);
  const lower = text.toLowerCase();
  const hit = words.map((word) => lower.indexOf(word)).filter((index) => index >= 0).sort((a, b) => a - b)[0];
  const start = Math.max(0, (hit ?? 0) - 80);
  const slice = text.slice(start, start + size).replace(/\s+/g, " ").trim();
  return `${start > 0 ? "…" : ""}${slice}${start + size < text.length ? "…" : ""}`;
}

function NormList({ filters, page, onPage, catalog }) {
  const norms = useNorms({ ...filters, offset: page * PAGE_SIZE, limit: PAGE_SIZE });
  const items = norms.data?.items || [];
  const total = norms.data?.total || 0;
  const pages = Math.ceil(total / PAGE_SIZE);
  if (norms.isLoading) return <p className="monitor-empty">Cargando normativa…</p>;
  if (norms.isError) return <p className="monitor-empty">No se pudo cargar la normativa.</p>;
  return (
    <>
      <p className="results-count">{total.toLocaleString("es-CO")} {total === 1 ? "norma" : "normas"}</p>
      {items.length ? <ul className="norm-results">
        {items.map((norm) => <li key={norm.id} className="norm-result">
          <div className="feed-item-meta">
            <span className="source-tag">{sourceName(catalog, norm.source)}</span>
            {norm.issued_on && <span>Expedida {formatDate(norm.issued_on)}</span>}
          </div>
          <Link className="feed-item-title" to={`/normativa/${norm.id}`}>{norm.label}</Link>
          {norm.title && <p className="feed-item-summary">{norm.title}</p>}
          <StatusBadge status={norm.status} />
        </li>)}
      </ul> : <p className="monitor-empty">No hay normas con estos filtros.</p>}
      {pages > 1 && <nav className="pagination" aria-label="Paginación">
        <button type="button" onClick={() => onPage(page - 1)} disabled={page === 0}>Anterior</button>
        <span>Página {page + 1} de {pages}</span>
        <button type="button" onClick={() => onPage(page + 1)} disabled={page + 1 >= pages}>Siguiente</button>
      </nav>}
    </>
  );
}

function TextResults({ query, catalog }) {
  const search = useCorpusSearch(query);
  if (query.trim().length < 2) return <p className="monitor-empty">Escribe un tema, una figura o una cita para buscar dentro de los artículos.</p>;
  if (search.isLoading) return <p className="monitor-empty">Buscando en los artículos…</p>;
  if (search.isError) return <p className="monitor-empty">No se pudo completar la búsqueda.</p>;
  const hits = search.data?.hits || [];
  if (!hits.length) return <p className="monitor-empty">Ningún artículo coincide con «{query}».</p>;
  return (
    <ul className="norm-results">
      {hits.map((hit) => <li key={`${hit.norm_id}-${hit.article_label}`} className="norm-result">
        <div className="feed-item-meta">
          <span className="source-tag">{sourceName(catalog, hit.source)}</span>
          <span>{hit.article_label}</span>
        </div>
        <Link className="feed-item-title" to={`/normativa/${hit.norm_id}#${encodeURIComponent(hit.article_label)}`}>{hit.norm_label}</Link>
        <p className="hit-excerpt">{excerpt(hit.content, query)}</p>
        <StatusBadge status={hit.status} />
      </li>)}
    </ul>
  );
}

export default function NormsPage() {
  const [params, setParams] = useSearchParams();
  const catalog = useSourceCatalog();
  const mode = params.get("mode") === "texto" ? "texto" : "titulo";
  const [query, setQuery] = useState(params.get("q") || "");
  const filters = {
    q: params.get("q") || "",
    source: params.get("source") || "",
    doc_type: params.get("doc_type") || "",
    status: params.get("status") || "",
    year: params.get("year") || "",
  };
  const page = Number(params.get("page") || 0);
  const currentYear = new Date().getFullYear();
  const years = Array.from({ length: currentYear - FIRST_YEAR + 1 }, (_, index) => currentYear - index);

  useEffect(() => {
    const term = query.trim();
    if (term === (params.get("q") || "")) return undefined;
    const timeout = window.setTimeout(() => {
      const next = new URLSearchParams(params);
      if (term) next.set("q", term);
      else next.delete("q");
      next.delete("page");
      setParams(next, { replace: true });
    }, 300);
    return () => window.clearTimeout(timeout);
  }, [query, params, setParams]);

  function update(changes) {
    const next = new URLSearchParams(params);
    Object.entries(changes).forEach(([key, value]) => {
      if (value) next.set(key, value);
      else next.delete(key);
    });
    setParams(next, { replace: true });
  }

  return (
    <div className="norms-page">
      <header className="corpus-hero compact">
        <div className="welcome-rule"><span /> NORMATIVA</div>
        <h1>Explora la regulación <em>vigente e histórica.</em></h1>
      </header>
      <div className="norms-search">
        <div className="mode-toggle" role="tablist" aria-label="Tipo de búsqueda">
          <button type="button" role="tab" aria-selected={mode === "titulo"} className={mode === "titulo" ? "is-active" : ""} onClick={() => update({ mode: "", page: "" })}>Título o número</button>
          <button type="button" role="tab" aria-selected={mode === "texto"} className={mode === "texto" ? "is-active" : ""} onClick={() => update({ mode: "texto", page: "" })}>Dentro del texto</button>
        </div>
        <label className="norms-search-box">
          <Icon name="search" size={16} />
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={mode === "texto" ? "Ej. medición bidireccional, excedentes AGPE, cargo por confiabilidad" : "Ej. 174 de 2021, autogeneración, comercialización"}
            aria-label={mode === "texto" ? "Buscar dentro del texto de las normas" : "Buscar por título o número"}
          />
        </label>
        {mode === "titulo" && <div className="norms-filters">
          <select aria-label="Fuente" value={filters.source} onChange={(event) => update({ source: event.target.value, doc_type: "", page: "" })}>
            <option value="">Todas las fuentes</option>
            {(catalog.data?.sources || []).map((source) => <option key={source.id} value={source.id}>{source.name}</option>)}
          </select>
          <select aria-label="Tipo de norma" value={filters.doc_type} onChange={(event) => update({ doc_type: event.target.value, page: "" })}>
            <option value="">Todos los tipos</option>
            {allDocTypes(catalog.data, filters.source).map((type) => <option key={type.id} value={type.id}>{type.label}</option>)}
          </select>
          <select aria-label="Estado" value={filters.status} onChange={(event) => update({ status: event.target.value, page: "" })}>
            <option value="">Cualquier estado</option>
            {Object.entries(STATUS).map(([id, info]) => <option key={id} value={id}>{info.label}</option>)}
          </select>
          <select aria-label="Año" value={filters.year} onChange={(event) => update({ year: event.target.value, page: "" })}>
            <option value="">Todos los años</option>
            {years.map((year) => <option key={year} value={year}>{year}</option>)}
          </select>
        </div>}
      </div>
      {mode === "texto"
        ? <TextResults query={filters.q} catalog={catalog.data} />
        : <NormList filters={filters} page={page} onPage={(next) => update({ page: next ? String(next) : "" })} catalog={catalog.data} />}
    </div>
  );
}
