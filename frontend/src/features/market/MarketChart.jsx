import { useEffect, useMemo, useRef, useState } from "react";

import { formatNumber, seriesColor } from "./format";

const DEFAULT_WIDTH = 640;
const PAD = { top: 12, right: 16, bottom: 28, left: 64 };

// Drawing at the container's real width keeps axis text at its CSS size on narrow screens.
function useWidth() {
  const ref = useRef(null);
  const [width, setWidth] = useState(DEFAULT_WIDTH);
  useEffect(() => {
    const element = ref.current;
    if (!element || typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(280, Math.round(entry.contentRect.width))));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

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

function isPresent(value) {
  return value !== null && value !== undefined;
}

function useGeometry(chart, type, height, width) {
  return useMemo(() => {
    const count = chart.x.length;
    const plotWidth = width - PAD.left - PAD.right;
    const plotHeight = height - PAD.top - PAD.bottom;
    const stacked = type === "stacked";
    const totals = stacked
      ? chart.x.map((_, index) => chart.series.reduce((sum, series) => sum + Math.max(0, series.values[index] || 0), 0))
      : [];
    const values = stacked ? totals : chart.series.flatMap((series) => series.values).filter(isPresent);
    if (!values.length) return null;
    // Bars grow from zero; lines need no zero baseline, so a reservoir moving from 79 % to 76 %
    // stays visible.
    const bars = type !== "line";
    const ticks = niceTicks(bars ? Math.min(0, ...values) : Math.min(...values), bars ? Math.max(0, ...values) : Math.max(...values));
    const low = ticks[0];
    const high = ticks.at(-1) === low ? low + 1 : ticks.at(-1);
    const band = plotWidth / Math.max(count, 1);
    const x = bars
      ? (index) => PAD.left + band * (index + 0.5)
      : (index) => PAD.left + (count === 1 ? plotWidth / 2 : (index / (count - 1)) * plotWidth);
    const y = (value) => PAD.top + plotHeight - ((value - low) / (high - low)) * plotHeight;
    return { ticks, x, y, band, plotWidth, bars, stacked, count, width };
  }, [chart, type, height, width]);
}

function Lines({ chart, geometry }) {
  const { x, y } = geometry;
  return chart.series.map((series, index) => {
    let path = "";
    series.values.forEach((value, position) => {
      if (!isPresent(value)) return;
      path += `${path && isPresent(series.values[position - 1]) ? "L" : "M"}${x(position).toFixed(1)},${y(value).toFixed(1)}`;
    });
    return <path key={series.name} d={path} className="market-line" style={{ stroke: seriesColor(series.name, index) }} />;
  });
}

function Bars({ chart, geometry }) {
  const { x, y, band, stacked } = geometry;
  const zero = y(Math.max(geometry.ticks[0], 0));
  const groups = stacked ? 1 : chart.series.length;
  const width = Math.max(1, Math.min(24, (band * 0.8) / groups));
  // A 2px surface gap separates neighbours only where the bars are wide enough to keep it.
  const gap = width > 6 ? 2 : 0;
  const rects = [];
  chart.x.forEach((_, position) => {
    let base = 0;
    chart.series.forEach((series, index) => {
      const value = series.values[position];
      if (!isPresent(value)) return;
      const left = stacked ? x(position) - width / 2 : x(position) - (width * groups) / 2 + width * index;
      const from = stacked ? base : 0;
      const to = stacked ? base + Math.max(0, value) : value;
      if (stacked) base = to;
      const top = Math.min(y(from), y(to));
      const barHeight = Math.max(stacked ? 0 : 1, Math.abs(y(to) - y(from)) - (stacked ? gap / 2 : 0));
      rects.push(<rect
        key={`${series.name}-${position}`}
        x={left + gap / 2}
        y={top}
        width={Math.max(1, width - gap)}
        height={barHeight}
        rx={width > 8 ? 2 : 0}
        style={{ fill: seriesColor(series.name, index) }}
      />);
    });
  });
  return <g>{rects}<line className="market-baseline" x1={PAD.left} x2={geometry.width - PAD.right} y1={zero} y2={zero} /></g>;
}

export default function MarketChart({ source, type = "line", height = 220, caption = true }) {
  const { chart } = source;
  const [hover, setHover] = useState(null);
  const [plotRef, measured] = useWidth();
  const geometry = useGeometry(chart, type, height, measured);
  if (!geometry) return <div className="market-chart is-empty">XM no devolvió datos para este periodo.</div>;
  const { width } = geometry;
  const { ticks, x, y, plotWidth, bars, count } = geometry;
  const multiple = chart.series.length > 1;

  function onMove(event) {
    const box = event.currentTarget.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, ((event.clientX - box.left) / box.width * width - PAD.left) / plotWidth));
    setHover(bars ? Math.min(count - 1, Math.floor(ratio * count)) : Math.round(ratio * (count - 1)));
  }

  const hoverLeft = hover === null ? 0 : Math.min(72, Math.max(2, (x(hover) / width) * 100));

  return (
    <figure className="market-chart">
      {caption && <figcaption>
        <span className="source-id">{source.id}</span>
        <strong>{source.title.replace(/^XM · /, "")}</strong>
        <small>{chart.unit} · fuente: XM</small>
      </figcaption>}
      {multiple && <ul className="market-legend">
        {chart.series.map((series, index) => <li key={series.name}><span style={{ background: seriesColor(series.name, index) }} />{series.name}</li>)}
      </ul>}
      <div className="market-plot" ref={plotRef}>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={`${source.title}. Gráfico de ${count} puntos en ${chart.unit}; los datos están en la tabla.`}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        >
          {ticks.map((tick) => <g key={tick}>
            <line className="market-grid" x1={PAD.left} x2={width - PAD.right} y1={y(tick)} y2={y(tick)} />
            <text className="market-axis" x={PAD.left - 8} y={y(tick) + 4} textAnchor="end">{formatNumber(tick)}</text>
          </g>)}
          <text className="market-axis" x={PAD.left} y={height - 8}>{chart.x[0]}</text>
          {count > 1 && <text className="market-axis" x={width - PAD.right} y={height - 8} textAnchor="end">{chart.x.at(-1)}</text>}
          {bars ? <Bars chart={chart} geometry={geometry} /> : <Lines chart={chart} geometry={geometry} />}
          {hover !== null && <g>
            <line className="market-crosshair" x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={height - PAD.bottom} />
            {!bars && chart.series.map((series, index) => isPresent(series.values[hover]) && <circle
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
            {multiple && `${series.name}: `}{isPresent(series.values[hover]) ? `${formatNumber(series.values[hover])} ${chart.unit}` : "—"}
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
              {chart.series.map((series) => <td key={series.name}>{isPresent(series.values[row]) ? formatNumber(series.values[row]) : "—"}</td>)}
            </tr>)}</tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}
