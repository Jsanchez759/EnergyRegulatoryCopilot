import { Outlet, useLocation } from "react-router-dom";

import AccountControls from "../auth/AccountControls";
import { useCorpusStatus, useSourceCatalog } from "./api";
import { formatDateTime, sourceName } from "./labels";
import Icon from "../../shared/Icon";
import PrimaryNav, { BrandLink } from "../../shared/PrimaryNav";
import useResizableWidth from "../../shared/useResizableWidth";
import useTheme from "../../shared/useTheme";

const SECTION_TITLES = { "/monitor": "Monitor regulatorio", "/normativa": "Normativa" };

function SourceHealth() {
  const catalog = useSourceCatalog();
  const status = useCorpusStatus();
  const bySource = new Map((status.data?.sources || []).map((item) => [item.source, item]));
  const sources = catalog.data?.sources || [];

  return (
    <section className="source-health" aria-label="Estado de las fuentes">
      <div className="sidebar-label">Fuentes</div>
      {sources.map((source) => {
        const health = bySource.get(source.id);
        const failing = health?.last_run_status === "failed" || (health?.last_run_errors || 0) > 0;
        return <div key={source.id} className="source-health-row" title={source.full_name}>
          <span className={`source-health-dot ${failing ? "is-failing" : health?.last_success_at ? "is-ok" : ""}`} />
          <div>
            <strong>{sourceName(catalog.data, source.id)}</strong>
            <small>{health ? `${health.indexed.toLocaleString("es-CO")} normas indexadas` : "Sin datos todavía"}</small>
            {health?.last_success_at && <small>Revisada {formatDateTime(health.last_success_at)}</small>}
          </div>
        </div>;
      })}
    </section>
  );
}

export default function CorpusLayout({ user }) {
  const location = useLocation();
  const [theme, setTheme] = useTheme();
  const sidebar = useResizableWidth({
    storageKey: "enercol_history_width_v1",
    defaultWidth: 260,
    minWidth: 200,
    maxWidth: 480,
  });
  const section = location.pathname.startsWith("/normativa") ? "/normativa" : "/monitor";

  return (
    <main
      className={`corpus-layout ${sidebar.isResizing ? "is-resizing" : ""}`}
      style={{ "--history-width": `${sidebar.width}px` }}
    >
      <aside className="history-pane" aria-label="Navegación">
        <BrandLink />
        <PrimaryNav />
        <SourceHealth />
        <div className="sidebar-bottom">
          <div className="sidebar-caption">Monitor regulatorio</div>
          <div className="sidebar-subcaption">Sector eléctrico · Colombia</div>
          <button className="theme-toggle" type="button" onClick={() => setTheme((value) => value === "light" ? "dark" : "light")}>
            <Icon name="sun" size={16} /> Apariencia <span>{theme === "light" ? "Claro" : "Oscuro"}</span>
          </button>
        </div>
        <div className="history-resize-handle" aria-label="Ajustar ancho del panel de navegación" title="Arrastra para ajustar · doble clic para restablecer" {...sidebar.handleProps} />
      </aside>
      <section className="corpus-main">
        <header className="topbar">
          <div className="breadcrumb"><span>Colombia</span><span className="breadcrumb-separator">/</span><strong>{SECTION_TITLES[section]}</strong></div>
          <div className="topbar-actions"><AccountControls user={user} /></div>
        </header>
        <div className="corpus-content"><Outlet /></div>
      </section>
    </main>
  );
}
