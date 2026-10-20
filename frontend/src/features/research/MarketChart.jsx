import { useMemo, useState } from "react";

const WIDTH = 640;
const HEIGHT = 220;
const PAD = { top: 12, right: 16, bottom: 28, left: 64 };
const OTHER = "Otros";

const numberFormat = new Intl.NumberFormat("es-CO", { maximumFractionDigits: 2 });

function niceTicks(min, max, count = 4) {
  if (min === max) {
    const pad = Math.abs(min) * 0.1 || 1;
    return niceTicks(min - pad, max + pad, count);
  }
  const raw = (max - min) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((factor) => factor * magnitude).find((value) => value >= raw);
  const ticks = [];
  for (let value = Math.floor(min / step) * step; value <= max + step / 2; value += step) ticks.push(value);
  return ticks;
}

function seriesColor(name, index) {
  return name === OTHER ? "var(--series-other)" : `var(--series-${index + 1})`;
}

export default function MarketChart({ source }) {
  const { chart } = source;
  const [hover, setHover] = useState(null);
  const geometry = useMemo(() => {
    const values = chart.series.flatMap((series) => series.values).filter((value) => value !== null && value !== undefined);
    if (!values.length) return null;
    // Lines need no zero baseline; a reservoir moving from 79 % to 76 % must stay visible.
    const ticks = niceTicks(Math.min(...values), Math.max(...values));
    const low = ticks[0];
    const high = ticks.at(-1) === low ? low + 1 : ticks.at(-1);
    const plotWidth = WIDTH - PAD.left - PAD.right;
    const plotHeight = HEIGHT - PAD.top - PAD.bottom;
    const x = (index) => PAD.left + (chart.x.length === 1 ? plotWidth / 2 : (index / (chart.x.length - 1)) * plotWidth);
    const y = (value) => PAD.top + plotHeight - ((value - low) / (high - low)) * plotHeight;
    const paths = chart.series.map((series) => {
      let path = "";
      series.values.forEach((value, index) => {
        if (value === null || value === undefined) return;
        path += `${path && series.values[index - 1] != null ? "L" : "M"}${x(index).toFixed(1)},${y(value).toFixed(1)}`;
      });
      return path;
    });
    return { ticks, x, y, paths, plotWidth };
  }, [chart]);

  if (!geometry) return null;
  const { ticks, x, y, paths, plotWidth } = geometry;
  const multiple = chart.series.length > 1;

  function onMove(event) {
    const box = event.currentTarget.getBoundingClientRect();
    const position = ((event.clientX - box.left) / box.width) * WIDTH;
    const ratio = Math.min(1, Math.max(0, (position - PAD.left) / plotWidth));
    setHover(Math.round(ratio * (chart.x.length - 1)));
  }

  const hoverLeft = hover === null ? 0 : Math.min(78, Math.max(4, (x(hover) / WIDTH) * 100));

  return (
    <figure className="market-chart">
      <figcaption>
        <span className="source-id">{source.id}</span>
        <strong>{source.title.replace(/^XM · /, "")}</strong>
        <small>{chart.unit} · fuente: XM</small>
      </figcaption>
      {multiple && <ul className="market-legend">
        {chart.series.map((series, index) => <li key={series.name}><span style={{ background: seriesColor(series.name, index) }} />{series.name}</li>)}
      </ul>}
      <div className="market-plot">
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          role="img"
          aria-label={`${source.title}. Gráfico de ${chart.x.length} puntos en ${chart.unit}; los datos están en la tabla.`}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        >
          {ticks.map((tick) => <g key={tick}>
            <line className="market-grid" x1={PAD.left} x2={WIDTH - PAD.right} y1={y(tick)} y2={y(tick)} />
            <text className="market-axis" x={PAD.left - 8} y={y(tick) + 4} textAnchor="end">{numberFormat.format(tick)}</text>
          </g>)}
          <text className="market-axis" x={PAD.left} y={HEIGHT - 8}>{chart.x[0]}</text>
          {chart.x.length > 1 && <text className="market-axis" x={WIDTH - PAD.right} y={HEIGHT - 8} textAnchor="end">{chart.x.at(-1)}</text>}
          {paths.map((path, index) => <path key={chart.series[index].name} d={path} className="market-line" style={{ stroke: seriesColor(chart.series[index].name, index) }} />)}
          {hover !== null && <g>
            <line className="market-crosshair" x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={HEIGHT - PAD.bottom} />
            {chart.series.map((series, index) => series.values[hover] != null && <circle
              key={series.name}
              className="market-dot"
              cx={x(hover)}
              cy={y(series.values[hover])}
              r="4"
              style={{ fill: seriesColor(series.name, index) }}
            />)}
          </g>}
        </svg>
        {hover !== null && <div className="market-tooltip" style={{ left: `${hoverLeft}%` }}>
          <strong>{chart.x[hover]}</strong>
          {chart.series.map((series, index) => <span key={series.name}>
            <i style={{ background: seriesColor(series.name, index) }} />
            {multiple && `${series.name}: `}{series.values[hover] == null ? "—" : `${numberFormat.format(series.values[hover])} ${chart.unit}`}
          </span>)}
        </div>}
      </div>
      <details className="market-table">
        <summary>Ver datos</summary>
        <div className="market-table-scroll">
          <table>
            <thead><tr><th>Periodo</th>{chart.series.map((series) => <th key={series.name}>{series.name}</th>)}</tr></thead>
            <tbody>{chart.x.map((label, row) => <tr key={label}>
              <td>{label}</td>
              {chart.series.map((series) => <td key={series.name}>{series.values[row] == null ? "—" : numberFormat.format(series.values[row])}</td>)}
            </tr>)}</tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}
