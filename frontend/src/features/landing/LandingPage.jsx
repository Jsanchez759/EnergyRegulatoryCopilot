import { Link } from "react-router-dom";

import { formatDate } from "../corpus/labels";
import { formatCompact } from "../market/format";
import { Change, Sparkline } from "../market/Overview";
import Icon from "../../shared/Icon";
import { UPCOMING_CORPUS, UPCOMING_MARKET } from "../../shared/sources";
import { usePublicOverview } from "./api";

const MODULES = [
  {
    icon: "book",
    name: "Investigación",
    title: "Pregunta y recibe la norma exacta",
    points: [
      "Respuestas con cita a la resolución, el artículo y su vigencia",
      "Cruza la regulación con tus propios PDFs",
      "Muestra qué buscó y qué fuentes usó",
    ],
  },
  {
    icon: "bell",
    name: "Monitor regulatorio",
    title: "No se te pasa una consulta",
    points: [
      "Proyectos de resolución y decreto apenas se publican",
      "Plazos de comentarios con días restantes",
      "Cambios de vigencia: derogatorias y modificaciones",
    ],
  },
  {
    icon: "chart",
    name: "Mercado",
    title: "Datos de XM, listos para graficar",
    points: [
      "Precio de bolsa, escasez, demanda, embalses y aportes",
      "Arma gráficas, compáralas y descárgalas en CSV",
      "Un asistente que consulta y grafica por ti",
    ],
  },
];

const STEPS = [
  { title: "Busca en la fuente oficial", detail: "Lee las normas de la CREG y el Ministerio de Minas y Energía, artículo por artículo, y los datos de XM en vivo." },
  { title: "Cita lo que encontró", detail: "Cada afirmación lleva su fuente: número de resolución, artículo y enlace al documento original." },
  { title: "Te deja verificar", detail: "Ves el método, las búsquedas y la evidencia. Si algo no está en las fuentes, lo dice." },
];

const AUDIENCES = [
  { name: "Comercializadores", detail: "Siguen los cambios a la fórmula tarifaria, las garantías y las reglas de mercado que afectan su margen." },
  { name: "Generadores", detail: "Vigilan cargo por confiabilidad, subastas y despacho, y cruzan la regulación con precios y embalses." },
  { name: "Consultoras y firmas legales", detail: "Preparan conceptos y comentarios a proyectos normativos con la cita exacta en minutos." },
  { name: "Asuntos regulatorios", detail: "Llevan el calendario de consultas y explican al resto de la empresa qué cambió y por qué." },
];

const FAQ = [
  { question: "¿De dónde salen los datos?", answer: "De fuentes oficiales: el gestor normativo de la CREG, el normograma y los foros del Ministerio de Minas y Energía, y la API pública de XM (Sinergox). EnerCol las revisa varias veces al día." },
  { question: "¿Puede equivocarse?", answer: "Sí, como cualquier asistente. Por eso cada respuesta enlaza a la norma o al dato de origen y muestra cómo se construyó: puedes verificar antes de usarla." },
  { question: "¿Qué pasa con mis documentos?", answer: "Los PDFs que subes quedan privados en tu cuenta. Solo tú puedes consultarlos y puedes borrarlos cuando quieras." },
  { question: "¿Cuánto cuesta?", answer: "Nada durante la beta. Aprobamos accesos por orden de llegada para mantener la calidad del servicio." },
];

function daysUntil(isoDate) {
  const end = new Date(`${isoDate}T12:00:00`);
  const today = new Date();
  today.setHours(12, 0, 0, 0);
  return Math.round((end - today) / 86_400_000);
}

function closingLabel(isoDate) {
  const days = daysUntil(isoDate);
  if (days <= 0) return "Cierra hoy";
  if (days === 1) return "Cierra mañana";
  return `Cierra en ${days} días`;
}

function LivePanel({ overview }) {
  const data = overview.data;
  return (
    <aside className="landing-live" aria-label="EnerCol en vivo">
      <div className="landing-live-head"><span className="landing-live-dot" /> En vivo desde las fuentes oficiales</div>
      <section className="landing-live-block">
        {data && !data.open_consultations
          ? <div className="landing-live-title is-latest"><strong>Últimas normas publicadas</strong><span>· sin consultas abiertas hoy</span></div>
          : <div className="landing-live-title">
            <strong>{overview.isLoading ? "—" : data?.open_consultations ?? 0}</strong>
            <span>consultas regulatorias abiertas a comentarios</span>
          </div>}
        {overview.isLoading && <div className="landing-live-skeleton" />}
        {data?.closing_soon?.length > 0 && <ul className="landing-consultations">
          {data.closing_soon.map((item) => <li key={item.url}>
            <a href={item.url} target="_blank" rel="noreferrer">{item.label}</a>
            <small>{item.title}</small>
            <span className={daysUntil(item.comment_end) <= 3 ? "is-urgent" : ""} title={`Comentarios hasta ${formatDate(item.comment_end)}`}>{closingLabel(item.comment_end)}</span>
          </li>)}
        </ul>}
        {data && !data.closing_soon?.length && data.latest?.length > 0 && <ul className="landing-consultations">
          {data.latest.map((item) => <li key={item.url}>
            <a href={item.url} target="_blank" rel="noreferrer">{item.label}</a>
            <small>{item.title}</small>
            <span>{formatDate(item.issued_on)}</span>
          </li>)}
        </ul>}
      </section>
      {data?.market?.length > 0 && <section className="landing-market" aria-label="Mercado eléctrico">
        {data.market.map((indicator) => <div key={indicator.key} className="landing-market-tile">
          <span className="market-tile-label">{indicator.label}</span>
          <span className="market-tile-value">{formatCompact(indicator.latest_value)}<small> {indicator.unit}</small></span>
          <Change indicator={indicator} />
          <Sparkline chart={indicator.spark} />
        </div>)}
        <small className="landing-market-source">Fuente: XM · dato del {data.market[0].latest_date}</small>
      </section>}
      {overview.isError && <p className="landing-live-empty">Los datos en vivo no están disponibles en este momento.</p>}
    </aside>
  );
}

function Coverage({ overview }) {
  const sources = overview.data?.sources || [];
  const norms = sources.reduce((sum, source) => sum + source.norms, 0);
  const figures = [
    { value: norms, label: "normas indexadas" },
    { value: overview.data?.articles, label: "artículos consultables" },
    // XM is live data rather than an indexed corpus, so it is not in the source list.
    { value: sources.length + 1, label: "fuentes oficiales activas" },
  ];
  return (
    <section className="landing-figures" aria-label="Cobertura">
      {figures.map((figure) => <div key={figure.label}>
        <strong>{overview.isLoading || figure.value === undefined ? "—" : figure.value.toLocaleString("es-CO")}</strong>
        <span>{figure.label}</span>
      </div>)}
    </section>
  );
}

function Sources({ overview }) {
  const sources = overview.data?.sources || [];
  return (
    <section className="landing-section" id="fuentes" aria-labelledby="fuentes-title">
      <div className="landing-section-head">
        <div className="welcome-rule"><span /> FUENTES</div>
        <h2 id="fuentes-title">Solo fuentes oficiales, <em>revisadas a diario.</em></h2>
        <p>EnerCol lee directamente a quienes regulan y operan el sector. La cobertura crece cada mes.</p>
      </div>
      <div className="landing-sources">
        {sources.map((source) => <div key={source.id} className="landing-source is-live">
          <span className="landing-source-state">Activa</span>
          <strong>{source.name}</strong>
          <small>{source.norms ? `${source.norms.toLocaleString("es-CO")} normas` : "Indexando"}</small>
        </div>)}
        <div className="landing-source is-live">
          <span className="landing-source-state">Activa</span>
          <strong>XM · Sinergox</strong>
          <small>Datos del mercado en vivo</small>
        </div>
        {[...UPCOMING_CORPUS, ...UPCOMING_MARKET].map((source) => <div key={source.name} className="landing-source">
          <span className="soon-badge">Pronto</span>
          <strong>{source.name}</strong>
          <small>{source.detail}</small>
        </div>)}
      </div>
    </section>
  );
}

export default function LandingPage() {
  const overview = usePublicOverview();
  return (
    <div className="landing">
      <header className="landing-nav">
        <Link className="brand" to="/" aria-label="EnerCol inicio">
          <span className="brand-mark"><span /><span /><span /></span><span>enercol<span className="brand-period">.</span></span>
        </Link>
        <nav aria-label="Secciones de la página">
          <a href="#producto">Producto</a>
          <a href="#fuentes">Fuentes</a>
          <a href="#para-quien">Para quién</a>
          <a href="#preguntas">Preguntas</a>
        </nav>
        <div className="landing-nav-actions">
          <Link className="landing-signin" to="/ingresar">Iniciar sesión</Link>
          <Link className="landing-cta is-small" to="/registro">Unirme a la beta</Link>
        </div>
      </header>

      <section className="landing-hero">
        <div className="landing-hero-copy">
          <div className="welcome-rule"><span /> BETA GRATUITA · SECTOR ELÉCTRICO COLOMBIANO</div>
          <h1>La regulación y el mercado, <em>con evidencia.</em></h1>
          <p>EnerCol es el copiloto de investigación regulatoria para el sector eléctrico de Colombia. Responde con la cita exacta de la CREG y el Ministerio de Minas y Energía, vigila las consultas abiertas y grafica los datos de XM.</p>
          <div className="landing-hero-actions">
            <Link className="landing-cta" to="/registro">Unirme a la lista de espera <Icon name="arrow" size={16} /></Link>
            <a className="landing-secondary" href="#producto">Ver qué hace</a>
          </div>
          <small className="landing-hero-note">Gratis durante la beta · aprobamos accesos por orden de llegada</small>
        </div>
        <LivePanel overview={overview} />
      </section>

      <Coverage overview={overview} />

      <section className="landing-section" id="producto" aria-labelledby="producto-title">
        <div className="landing-section-head">
          <div className="welcome-rule"><span /> PRODUCTO</div>
          <h2 id="producto-title">Tres herramientas, <em>una sola fuente de verdad.</em></h2>
        </div>
        <div className="landing-modules">
          {MODULES.map((module) => <article key={module.name} className="landing-module">
            <span className="landing-module-icon"><Icon name={module.icon} size={20} /></span>
            <span className="landing-module-name">{module.name}</span>
            <h3>{module.title}</h3>
            <ul>{module.points.map((point) => <li key={point}><Icon name="check" size={14} />{point}</li>)}</ul>
          </article>)}
        </div>
      </section>

      <section className="landing-section landing-method" aria-labelledby="metodo-title">
        <div className="landing-section-head">
          <div className="welcome-rule"><span /> CÓMO RESPONDE</div>
          <h2 id="metodo-title">No es un chat genérico. <em>Es investigación verificable.</em></h2>
        </div>
        <ol className="landing-steps">
          {STEPS.map((step, index) => <li key={step.title}>
            <span>{String(index + 1).padStart(2, "0")}</span>
            <strong>{step.title}</strong>
            <p>{step.detail}</p>
          </li>)}
        </ol>
      </section>

      <Sources overview={overview} />

      <section className="landing-section" id="para-quien" aria-labelledby="audiencia-title">
        <div className="landing-section-head">
          <div className="welcome-rule"><span /> PARA QUIÉN</div>
          <h2 id="audiencia-title">Hecho para quienes <em>viven la regulación.</em></h2>
        </div>
        <div className="landing-audiences">
          {AUDIENCES.map((audience) => <div key={audience.name}>
            <strong>{audience.name}</strong>
            <p>{audience.detail}</p>
          </div>)}
        </div>
      </section>

      <section className="landing-section" id="preguntas" aria-labelledby="preguntas-title">
        <div className="landing-section-head">
          <div className="welcome-rule"><span /> PREGUNTAS</div>
          <h2 id="preguntas-title">Lo que suelen <em>preguntarnos.</em></h2>
        </div>
        <div className="landing-faq">
          {FAQ.map((item) => <details key={item.question}>
            <summary>{item.question}</summary>
            <p>{item.answer}</p>
          </details>)}
        </div>
      </section>

      <section className="landing-final" aria-labelledby="final-title">
        <h2 id="final-title">Entra a la beta <em>gratuita.</em></h2>
        <p>Déjanos tus datos y te damos acceso por orden de llegada.</p>
        <Link className="landing-cta" to="/registro">Unirme a la lista de espera <Icon name="arrow" size={16} /></Link>
      </section>

      <footer className="landing-footer">
        <span>enercol<span className="brand-period">.</span> · Sector eléctrico · Colombia</span>
        <span>EnerCol no es una fuente oficial. Verifica siempre en el documento original.</span>
      </footer>
    </div>
  );
}
