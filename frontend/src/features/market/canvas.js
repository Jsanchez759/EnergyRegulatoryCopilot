import { daysAgo, formatNumber, isoDate } from "./format";

export const RANGES = [
  { id: "7d", label: "7 días", days: 7 },
  { id: "30d", label: "30 días", days: 30 },
  { id: "90d", label: "90 días", days: 90 },
  { id: "1y", label: "1 año", days: 365 },
];

export function makeQuery({ metricId, entity = "Sistema", start, end, codes = [], groupByTechnology = false }) {
  return {
    metric_id: metricId,
    entity,
    start,
    end,
    codes: [...codes].sort(),
    group_by_technology: groupByTechnology,
  };
}

export function queryKey(query) {
  return [query.metric_id, query.entity, query.start, query.end, query.codes.join(","), query.group_by_technology].join("|");
}

export function rangeQuery(metricId, days, entity = "Sistema", today = new Date()) {
  return makeQuery({ metricId, entity, start: daysAgo(days, today), end: isoDate(today) });
}

export function defaultCanvas(today = new Date()) {
  return [
    {
      id: "inicio-precios",
      type: "line",
      layers: [rangeQuery("PPPrecBolsNaci", 30, "Sistema", today), rangeQuery("PrecEsca", 30, "Sistema", today)],
    },
    {
      id: "inicio-generacion",
      type: "stacked",
      layers: [{ ...rangeQuery("Gene", 14, "Recurso", today), group_by_technology: true }],
    },
  ];
}

function layerName(series, data, layerCount) {
  const metric = data.metric.name.replace(/ por (Sistema|Recurso|Agente|Embalse|Rio|Río)$/i, "");
  if (layerCount === 1) return series.name === "Sistema" ? metric : series.name;
  return series.name === "Sistema" ? metric : `${metric}: ${series.name}`;
}

/** Union of every layer's periods, with each layer's series aligned to it. */
export function mergeLayers(results) {
  const loaded = results.filter((result) => result?.chart);
  if (!loaded.length) return null;
  const x = [...new Set(loaded.flatMap((result) => result.chart.x))].sort();
  const series = loaded.flatMap((result) => {
    const position = new Map(result.chart.x.map((label, index) => [label, index]));
    return result.chart.series.map((item) => ({
      name: layerName(item, result, loaded.length),
      values: x.map((label) => (position.has(label) ? item.values[position.get(label)] : null)),
    }));
  });
  const units = [...new Set(loaded.map((result) => result.chart.unit))];
  const names = [...new Set(loaded.map((result) => result.metric.name))];
  const first = loaded[0];
  return {
    title: names.join(" vs. "),
    period: `${first.start} a ${first.end}`,
    unit: units.join(" / "),
    chart: { unit: units.join(" / "), x, series },
  };
}

export function seriesSummary(chart) {
  return chart.series.map((series) => {
    const points = chart.x.map((label, index) => [label, series.values[index]]).filter(([, value]) => value !== null && value !== undefined);
    if (!points.length) return { name: series.name, empty: true };
    const values = points.map(([, value]) => value);
    const low = points.reduce((best, point) => (point[1] < best[1] ? point : best));
    const high = points.reduce((best, point) => (point[1] > best[1] ? point : best));
    return {
      name: series.name,
      average: values.reduce((sum, value) => sum + value, 0) / values.length,
      low,
      high,
      last: points.at(-1),
    };
  });
}

/** Plain-text description of a chart for the assistant. */
export function describeChart(merged) {
  const lines = [`${merged.title} (${merged.period}, ${merged.unit})`];
  seriesSummary(merged.chart).forEach((item) => {
    if (item.empty) {
      lines.push(`- ${item.name}: sin datos`);
      return;
    }
    lines.push(
      `- ${item.name}: promedio ${formatNumber(item.average)}, mínimo ${formatNumber(item.low[1])} (${item.low[0]}), `
      + `máximo ${formatNumber(item.high[1])} (${item.high[0]}), último ${formatNumber(item.last[1])} (${item.last[0]})`,
    );
  });
  return lines.join("\n");
}

export function toCsv(merged) {
  const escape = (value) => (/[",\n;]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value);
  const header = ["periodo", ...merged.chart.series.map((series) => series.name)].map(escape).join(",");
  const rows = merged.chart.x.map((label, index) => [label, ...merged.chart.series.map((series) => series.values[index] ?? "")].map((value) => escape(String(value))).join(","));
  return [header, ...rows].join("\n");
}

/** Canvas charts for the series the assistant fetched; comparable series share a chart. */
export function chartsFromSources(sources, makeId) {
  const groups = new Map();
  (sources || []).filter((source) => source.kind === "market" && source.chart?.query?.metric_id).forEach((source) => {
    const key = `${source.chart.unit}|${source.chart.x.join(",")}`;
    groups.set(key, [...(groups.get(key) || []), source]);
  });
  return [...groups.values()].map((group) => {
    const layers = group.map((source) => makeQuery({
      metricId: source.chart.query.metric_id,
      entity: source.chart.query.entity,
      start: source.chart.query.start,
      end: source.chart.query.end,
      codes: source.chart.query.codes || [],
      groupByTechnology: source.chart.query.group_by_technology,
    }));
    return { id: makeId(), type: layers[0].group_by_technology ? "stacked" : "line", layers, fromAssistant: true };
  });
}
