import { useEffect, useMemo, useState } from "react";

import { useMarketCatalog, useMarketCodes, useMetricSearch } from "./api";
import { makeQuery, RANGES } from "./canvas";
import { daysAgo, isoDate } from "./format";
import Icon from "../../shared/Icon";

const ENTITY_LABELS = {
  Sistema: "Sistema (total nacional)",
  Recurso: "Planta",
  Agente: "Agente",
  Embalse: "Embalse",
  Rio: "Río",
};
const CODED_ENTITIES = new Set(["Recurso", "Agente", "Embalse", "Rio"]);
const GRANULARITY = { hourly: "horaria", daily: "diaria", monthly: "mensual" };

function useDebounced(value, delay = 250) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timeout = window.setTimeout(() => setDebounced(value), delay);
    return () => window.clearTimeout(timeout);
  }, [value, delay]);
  return debounced;
}

function MetricPicker({ catalog, selected, onSelect }) {
  const [text, setText] = useState("");
  const [open, setOpen] = useState(false);
  const query = useDebounced(text);
  const search = useMetricSearch(query);
  const recommended = catalog.filter((metric) => metric.recommended);
  const searching = text.trim().length >= 2;
  // Results for an earlier query would let a quick click pick a metric the user did not mean.
  const current = searching && query === text && !search.isPlaceholderData;
  const options = searching ? (current ? search.data?.metrics || [] : []) : recommended;

  return (
    <div className="market-field market-metric-picker">
      <label htmlFor="market-metric">Métrica</label>
      <div className="market-search">
        <Icon name="search" size={15} />
        <input
          id="market-metric"
          value={open ? text : selected ? `${selected.name}` : text}
          placeholder="Busca: precio de bolsa, embalses, generación solar…"
          onFocus={() => { setOpen(true); setText(""); }}
          onBlur={() => window.setTimeout(() => setOpen(false), 150)}
          onChange={(event) => setText(event.target.value)}
          autoComplete="off"
          role="combobox"
          aria-expanded={open}
          aria-controls="market-metric-options"
        />
      </div>
      {open && <ul id="market-metric-options" className="market-options" role="listbox">
        {!options.length && <li className="market-option-empty">{searching && (!current || search.isFetching) ? "Buscando…" : "Sin resultados"}</li>}
        {options.map((metric) => <li key={`${metric.metric_id}-${metric.entity}`}>
          <button type="button" role="option" aria-selected={false} onMouseDown={(event) => event.preventDefault()} onClick={() => { onSelect(metric); setOpen(false); }}>
            <strong>{metric.name}</strong>
            <small>{metric.metric_id} · {metric.units || "sin unidad"} · {GRANULARITY[metric.granularity]}</small>
            {metric.recommended && <span className="market-option-hint">{metric.recommended}</span>}
          </button>
        </li>)}
      </ul>}
    </div>
  );
}

function CodePicker({ entity, codes, onChange }) {
  const [text, setText] = useState("");
  const query = useDebounced(text);
  const found = useMarketCodes(entity, query);
  const options = (found.data?.codes || []).filter((item) => !codes.some((code) => code.code === item.code));
  return (
    <div className="market-field">
      <label htmlFor="market-codes">{ENTITY_LABELS[entity]}s</label>
      <div className="market-chips">
        {codes.map((code) => <span key={code.code} className="market-chip">
          {code.name} <small>{code.code}</small>
          <button type="button" aria-label={`Quitar ${code.name}`} onClick={() => onChange(codes.filter((item) => item.code !== code.code))}>×</button>
        </span>)}
      </div>
      <input id="market-codes" value={text} onChange={(event) => setText(event.target.value)} placeholder="Escribe un nombre: Ituango, Guavio, EPM…" autoComplete="off" />
      {text.trim().length >= 2 && options.length > 0 && <ul className="market-options is-inline">
        {options.slice(0, 6).map((item) => <li key={item.code}>
          <button type="button" onClick={() => { onChange([...codes, item].slice(0, 8)); setText(""); }}>
            <strong>{item.name}</strong><small>{item.code} · {item.detail}</small>
          </button>
        </li>)}
      </ul>}
    </div>
  );
}

export default function ChartBuilder({ charts, onCreate, onAddLayer }) {
  const catalog = useMarketCatalog();
  const metrics = useMemo(() => catalog.data?.metrics || [], [catalog.data]);
  const [metric, setMetric] = useState(null);
  const [entity, setEntity] = useState("Sistema");
  const [codes, setCodes] = useState([]);
  const [byTechnology, setByTechnology] = useState(false);
  const [range, setRange] = useState("30d");
  const [start, setStart] = useState(daysAgo(30));
  const [end, setEnd] = useState(isoDate(new Date()));
  const [target, setTarget] = useState("new");

  const entities = metric ? metrics.filter((item) => item.metric_id === metric.metric_id).map((item) => item.entity) : [];
  const chosen = metric && metrics.find((item) => item.metric_id === metric.metric_id && item.entity === entity);
  const compatible = charts.filter((chart) => chart.baseUnit === undefined || chart.baseUnit === chosen?.units);

  function selectMetric(next) {
    setMetric(next);
    setEntity(next.entity);
    setCodes([]);
    setByTechnology(false);
  }

  function selectRange(id) {
    const preset = RANGES.find((item) => item.id === id);
    setRange(id);
    setStart(daysAgo(preset.days));
    setEnd(isoDate(new Date()));
  }

  function submit(event) {
    event.preventDefault();
    if (!chosen) return;
    const query = makeQuery({
      metricId: chosen.metric_id,
      entity,
      start,
      end,
      codes: codes.map((code) => code.code),
      groupByTechnology: entity === "Recurso" && byTechnology,
    });
    if (target === "new") onCreate(query, chosen.units, byTechnology ? "stacked" : "line");
    else onAddLayer(target, query);
  }

  return (
    <form className="market-builder" onSubmit={submit} aria-label="Crear gráfica">
      <div className="market-builder-head">
        <h2>Crear gráfica</h2>
        <small>{metrics.length ? `${metrics.length} métricas de XM disponibles` : "Cargando catálogo…"}</small>
      </div>
      <div className="market-builder-grid">
        <MetricPicker catalog={metrics} selected={metric} onSelect={selectMetric} />
        <div className="market-field">
          <label htmlFor="market-entity">Desagregación</label>
          <select id="market-entity" value={entity} disabled={!metric} onChange={(event) => { setEntity(event.target.value); setCodes([]); setByTechnology(false); }}>
            {(entities.length ? entities : ["Sistema"]).map((item) => <option key={item} value={item}>{ENTITY_LABELS[item] || item}</option>)}
          </select>
        </div>
        <div className="market-field">
          <span className="market-label">Periodo</span>
          <div className="chip-group" role="group" aria-label="Periodo">
            {RANGES.map((item) => <button key={item.id} type="button" className={`chip ${range === item.id ? "is-active" : ""}`} aria-pressed={range === item.id} onClick={() => selectRange(item.id)}>{item.label}</button>)}
          </div>
          <div className="market-dates">
            <input type="date" aria-label="Desde" value={start} max={end} onChange={(event) => { setStart(event.target.value); setRange(""); }} />
            <span>a</span>
            <input type="date" aria-label="Hasta" value={end} min={start} onChange={(event) => { setEnd(event.target.value); setRange(""); }} />
          </div>
        </div>
      </div>
      {chosen && <p className="market-metric-note">{chosen.description || chosen.name} <span>· {chosen.units || "sin unidad"} · resolución {GRANULARITY[chosen.granularity]}</span></p>}
      {chosen && CODED_ENTITIES.has(entity) && <div className="market-builder-grid">
        {!byTechnology && <CodePicker entity={entity} codes={codes} onChange={setCodes} />}
        {entity === "Recurso" && <label className="market-toggle">
          <input type="checkbox" checked={byTechnology} onChange={(event) => setByTechnology(event.target.checked)} />
          Agrupar por tecnología (hidráulica, térmica, solar, eólica…)
        </label>}
        {!codes.length && !byTechnology && <p className="market-hint">Sin filtro se traen todas las entidades: limita el periodo a 31 días o elige algunas.</p>}
      </div>}
      <div className="market-builder-actions">
        <select aria-label="Destino" value={target} onChange={(event) => setTarget(event.target.value)}>
          <option value="new">Nueva gráfica</option>
          {compatible.map((chart, index) => <option key={chart.id} value={chart.id}>Añadir a la gráfica {index + 1}{chart.label ? `: ${chart.label}` : ""}</option>)}
        </select>
        <button type="submit" className="primary-action" disabled={!chosen}>Graficar</button>
      </div>
    </form>
  );
}
