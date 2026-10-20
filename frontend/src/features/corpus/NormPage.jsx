import { useEffect, useMemo } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";

import { fileUrl, useNorm, useSourceCatalog } from "./api";
import {
  consultation,
  docTypeLabel,
  EVENT_KINDS,
  formatDate,
  formatDateTime,
  relationVerb,
  sourceName,
  STATUS,
} from "./labels";
import StatusBadge from "./StatusBadge";

function baseLabel(label) {
  return label.replace(/\s*\(parte \d+\)$/, "");
}

function groupArticles(articles) {
  const groups = [];
  articles.forEach((article) => {
    const label = baseLabel(article.label);
    const last = groups.at(-1);
    if (last?.label === label) last.content += `\n${article.content}`;
    else groups.push({ label, content: article.content });
  });
  return groups;
}

// Vigencia notes such as "<Artículo modificado por …>" are shown apart from the rule itself.
function ArticleText({ text }) {
  return text.split("\n").filter(Boolean).map((paragraph, index) => (
    <p key={index}>
      {paragraph.split(/(<[^<>]{5,800}>)/).map((part, partIndex) => (
        part.startsWith("<") && part.endsWith(">")
          ? <span key={partIndex} className="vigencia-note">{part.slice(1, -1)}</span>
          : part
      ))}
    </p>
  ));
}

function RelationList({ title, relations, emptyText, incoming }) {
  return (
    <section className="norm-side-section">
      <h2>{title}</h2>
      {relations.length ? <ul className="relation-list">
        {relations.map((relation, index) => {
          const other = relation.other_norm_id
            ? <Link to={`/normativa/${relation.other_norm_id}`}>{relation.other_norm_label}</Link>
            : <span title="Norma aún no incorporada al corpus">{relation.other_norm_label}</span>;
          const fromArticle = relation.from_article ? `${relation.from_article} de ` : "";
          return <li key={`${relation.from_key}-${relation.to_article}-${index}`}>
            {incoming
              ? <>{fromArticle}{other} {relationVerb(relation.relation)} {relation.to_article || "la norma completa"}</>
              : <>{relation.from_article || "Esta norma"} {relationVerb(relation.relation)} {relation.to_article ? `${relation.to_article} de ` : ""}{other}</>}
          </li>;
        })}
      </ul> : <p className="monitor-empty">{emptyText}</p>}
    </section>
  );
}

export default function NormPage() {
  const { normId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const catalog = useSourceCatalog();
  const norm = useNorm(normId);
  const data = norm.data;
  const articles = useMemo(() => groupArticles(data?.articles || []), [data]);
  const amendmentsByArticle = useMemo(() => {
    const map = new Map();
    (data?.amended_by || []).forEach((relation) => {
      const key = relation.to_article || "";
      map.set(key, [...(map.get(key) || []), relation]);
    });
    return map;
  }, [data]);

  useEffect(() => {
    if (!data || !location.hash) return;
    document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView({ block: "start" });
  }, [data, location.hash]);

  if (norm.isLoading) return <p className="monitor-empty">Cargando norma…</p>;
  if (norm.isError || !data) return <div className="norm-missing"><p>No encontramos esta norma.</p><Link to="/normativa">Volver a la normativa</Link></div>;

  const original = data.files.find((file) => file.kind === "original" && file.stored);
  const wholeNormChanges = amendmentsByArticle.get("") || [];
  const comments = consultation(data);

  return (
    <article className="norm-page">
      <Link className="back-link" to="/normativa">← Normativa</Link>
      <header className="norm-header">
        <div className="norm-eyebrow">{sourceName(catalog.data, data.source)} · {docTypeLabel(catalog.data, data.source, data.doc_type)}</div>
        <h1>{data.label}</h1>
        {data.title && <p className="norm-title">{data.title}</p>}
        <dl className="norm-meta">
          <div><dt>Estado</dt><dd><StatusBadge status={data.status} /></dd></div>
          {data.issued_on && <div><dt>Expedida</dt><dd>{formatDate(data.issued_on)}</dd></div>}
          {data.published_on && <div><dt>Publicada</dt><dd>{formatDate(data.published_on)}</dd></div>}
          {comments && <div><dt>Comentarios</dt><dd>{formatDate(data.comment_start)} – {formatDate(data.comment_end)}</dd></div>}
          {data.diario_oficial && <div><dt>Diario Oficial</dt><dd>No. {data.diario_oficial}</dd></div>}
          {data.last_changed_at && <div><dt>Última actualización</dt><dd>{formatDateTime(data.last_changed_at)}</dd></div>}
        </dl>
        <div className="norm-actions">
          <button type="button" className="primary-action" onClick={() => navigate("/", { state: { draft: `Explícame en detalle la ${data.label}: qué establece, a quién aplica y qué obligaciones crea.` } })}>Preguntar al asistente</button>
          <a href={data.url} target="_blank" rel="noreferrer">Ver en la fuente oficial</a>
          {original && <a href={fileUrl(original.id)} target="_blank" rel="noreferrer">Documento original</a>}
        </div>
        {data.status === "derogada" && <p className="norm-notice is-derogada">{STATUS.derogada.hint}. Consulta las normas que la reemplazan antes de aplicarla.</p>}
        {data.status === "proyecto" && <p className="norm-notice is-proyecto">
          Proyecto publicado para comentarios: no es una norma vigente.{" "}
          {comments
            ? comments.open ? `Recibe comentarios hasta el ${formatDate(data.comment_end)}.` : `La consulta cerró el ${formatDate(data.comment_end)}.`
            : "El plazo de comentarios está en su encabezado."}
        </p>}
        {wholeNormChanges.length > 0 && <p className="norm-notice">{wholeNormChanges.map((item) => `${item.other_norm_label} ${relationVerb(item.relation)} esta norma`).join(" · ")}</p>}
      </header>

      <div className="norm-body">
        <nav className="article-index" aria-label="Índice de artículos">
          <h2>Contenido</h2>
          <ol>{articles.map((article) => <li key={article.label}>
            <a href={`#${encodeURIComponent(article.label)}`}>{article.label}</a>
            {amendmentsByArticle.has(article.label) && <span className="index-mark" title="Tiene modificaciones">•</span>}
          </li>)}</ol>
        </nav>

        <div className="article-text">
          {articles.map((article) => <section key={article.label} id={encodeURIComponent(article.label)} className="norm-article">
            <h2>{article.label}</h2>
            {amendmentsByArticle.has(article.label) && <ul className="article-amendments">
              {amendmentsByArticle.get(article.label).map((relation, index) => <li key={index}>
                {relation.other_norm_id ? <Link to={`/normativa/${relation.other_norm_id}`}>{relation.other_norm_label}</Link> : relation.other_norm_label}
                {" "}{relationVerb(relation.relation)} este artículo{relation.from_article ? ` (${relation.from_article})` : ""}
              </li>)}
            </ul>}
            <ArticleText text={article.content} />
          </section>)}
        </div>

        <aside className="norm-side">
          <RelationList title="Modificada por" relations={data.amended_by} emptyText="La fuente no registra modificaciones." incoming />
          <RelationList title="Esta norma modifica" relations={data.amends} emptyText="No se registran modificaciones a otras normas." />
          <section className="norm-side-section">
            <h2>Archivos</h2>
            {data.files.length ? <ul className="file-list">
              {data.files.map((file) => <li key={file.id}>
                <a href={file.stored ? fileUrl(file.id) : file.url} target="_blank" rel="noreferrer">{file.label}</a>
                <small>{file.kind === "original" ? "Original" : file.kind.toUpperCase()}</small>
              </li>)}
            </ul> : <p className="monitor-empty">Sin archivos asociados.</p>}
          </section>
          <section className="norm-side-section">
            <h2>Historial</h2>
            <ul className="history-list">
              {data.events.map((event) => <li key={event.id}><strong>{EVENT_KINDS[event.kind] || event.kind}</strong><small>{formatDateTime(event.occurred_at)}</small></li>)}
            </ul>
          </section>
        </aside>
      </div>
    </article>
  );
}
