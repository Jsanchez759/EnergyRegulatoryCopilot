import { useMarketOverview } from "./api";
import { formatCompact } from "./format";

function Sparkline({ chart }) {
  const values = chart.series[0]?.values || [];
  const points = values.map((value, index) => [index, value]).filter(([, value]) => value !== null && value !== undefined);
  if (points.length < 2) return null;
  const numbers = points.map(([, value]) => value);
  const low = Math.min(...numbers);
  const high = Math.max(...numbers);
  const span = high - low || 1;
  const path = points.map(([index, value], position) => `${position ? "L" : "M"}${(index / (values.length - 1)) * 100},${28 - ((value - low) / span) * 24}`).join("");
  return <svg className="market-spark" viewBox="0 0 100 30" preserveAspectRatio="none" aria-hidden="true"><path d={path} /></svg>;
}

function Change({ indicator }) {
  if (indicator.change === null || indicator.change === undefined) return null;
  const rounded = Math.round(indicator.change * 10) / 10;
  const arrow = rounded > 0 ? "▲" : rounded < 0 ? "▼" : "■";
  const unit = indicator.change_unit === "pp" ? " pp" : " %";
  return <small className="market-tile-change">{arrow} {rounded > 0 ? "+" : ""}{formatCompact(rounded)}{unit} en 7 días</small>;
}

export default function Overview({ onOpen }) {
  const overview = useMarketOverview();
  if (overview.isError) return <p className="monitor-empty">No pudimos cargar los indicadores de XM.</p>;
  const indicators = overview.data?.indicators || [];
  return (
    <section className="market-overview" aria-label="Indicadores del mercado">
      {overview.isLoading
        ? Array.from({ length: 5 }, (_, index) => <div key={index} className="market-tile is-loading" />)
        : indicators.map((indicator) => <button
          key={indicator.key}
          type="button"
          className="market-tile"
          onClick={() => onOpen(indicator)}
          title="Abrir en el lienzo con los últimos 90 días"
        >
          <span className="market-tile-label">{indicator.label}</span>
          <span className="market-tile-value">{formatCompact(indicator.latest_value)}<small> {indicator.unit}</small></span>
          <Change indicator={indicator} />
          <Sparkline chart={indicator.spark} />
          <small className="market-tile-date">Dato del {indicator.latest_date}</small>
        </button>)}
    </section>
  );
}
