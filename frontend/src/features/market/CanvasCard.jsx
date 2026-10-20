import { useEffect, useMemo } from "react";
import { useQueries } from "@tanstack/react-query";

import { seriesQueryOptions } from "./api";
import { mergeLayers, seriesSummary, toCsv } from "./canvas";
import { formatNumber } from "./format";
import { seriesColor } from "./format";
import MarketChart from "./MarketChart";

// Excel reads UTF-8 CSV correctly only with a byte-order mark.
const BOM = String.fromCharCode(0xfeff);
const TYPES = [
  { id: "line", label: "Líneas" },
  { id: "bar", label: "Barras" },
  { id: "stacked", label: "Apiladas" },
];

function download(merged) {
  const blob = new Blob([BOM + toCsv(merged)], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${merged.title.replace(/[^\p{L}\p{N}]+/gu, "-").toLowerCase()}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function layerLabel(layer, result) {
  const name = result?.metric?.name || layer.metric_id;
  const detail = layer.group_by_technology ? "por tecnología" : layer.codes.length ? layer.codes.join(", ") : "";
  return `${name}${detail ? ` · ${detail}` : ""} · ${layer.start} a ${layer.end}`;
}

export default function CanvasCard({ chart, index, onUpdate, onRemove, onRemoveLayer, onAsk, highlighted }) {
  const results = useQueries({ queries: chart.layers.map(seriesQueryOptions) });
  const data = results.map((result) => result.data);
  const merged = useMemo(() => mergeLayers(data), [data]);
  const loading = results.some((result) => result.isLoading);
  const errors = results.filter((result) => result.isError);
  const summaries = merged ? seriesSummary(merged.chart).slice(0, 4) : [];
  const title = merged?.title;

  useEffect(() => {
    if (title && chart.label !== title) onUpdate(chart.id, { label: title });
  }, [title, chart.id, chart.label, onUpdate]);

  return (
    <article className={`market-card ${highlighted ? "is-new" : ""}`} id={`grafica-${chart.id}`} aria-label={`Gráfica ${index + 1}`}>
      <header className="market-card-head">
        <div>
          <span className="market-card-index">Gráfica {index + 1}{chart.fromAssistant ? " · del asistente" : ""}</span>
          <h3>{merged?.title || "Cargando datos de XM…"}</h3>
          {merged && <small>{merged.period} · {merged.unit}</small>}
        </div>
        <div className="market-card-actions">
          <div className="mode-toggle" role="tablist" aria-label="Tipo de gráfica">
            {TYPES.filter((type) => type.id !== "stacked" || (merged?.chart.series.length || 0) > 1).map((type) => <button
              key={type.id}
              type="button"
              role="tab"
              aria-selected={chart.type === type.id}
              className={chart.type === type.id ? "is-active" : ""}
              onClick={() => onUpdate(chart.id, { type: type.id })}
            >{type.label}</button>)}
          </div>
          <button type="button" className="market-card-button" disabled={!merged} onClick={() => onAsk(chart, merged)}>Preguntar</button>
          <button type="button" className="market-card-button" disabled={!merged} onClick={() => download(merged)}>CSV</button>
          <button type="button" className="market-card-button is-quiet" aria-label={`Quitar gráfica ${index + 1}`} onClick={() => onRemove(chart.id)}>×</button>
        </div>
      </header>
      <ul className="market-layers">
        {chart.layers.map((layer, position) => <li key={`${layer.metric_id}-${position}`}>
          <span>{layerLabel(layer, data[position])}</span>
          {chart.layers.length > 1 && <button type="button" aria-label="Quitar serie" onClick={() => onRemoveLayer(chart.id, position)}>×</button>}
        </li>)}
      </ul>
      {errors.map((result, position) => <p key={position} className="market-card-error">{result.error.message}</p>)}
      {loading && !merged && <div className="market-card-loading"><span /><span /><span /> Consultando XM</div>}
      {merged && <>
        <MarketChart source={{ id: "", title: merged.title, chart: merged.chart }} type={chart.type} height={260} caption={false} />
        <dl className="market-stats">
          {summaries.map((item, position) => <div key={item.name}>
            <dt><i style={{ background: seriesColor(item.name, position) }} />{item.name}</dt>
            {item.empty ? <dd>Sin datos en el periodo</dd> : <dd>
              <span>Promedio <strong>{formatNumber(item.average)}</strong></span>
              <span>Mín. <strong>{formatNumber(item.low[1])}</strong> <small>{item.low[0]}</small></span>
              <span>Máx. <strong>{formatNumber(item.high[1])}</strong> <small>{item.high[0]}</small></span>
              <span>Último <strong>{formatNumber(item.last[1])}</strong> <small>{item.last[0]}</small></span>
            </dd>}
          </div>)}
        </dl>
      </>}
    </article>
  );
}
