function metricName(source) {
  return source.title.replace(/^XM · /, "").split(" · ")[0];
}

// Series that share unit and dates read best on one chart, e.g. spot price against scarcity price.
export function combineMarketSources(sources) {
  const groups = new Map();
  sources.filter((source) => source.kind === "market" && source.chart?.x?.length).forEach((source) => {
    const key = `${source.chart.unit}|${source.chart.x.join(",")}`;
    groups.set(key, [...(groups.get(key) || []), source]);
  });
  return [...groups.values()].map((group) => {
    if (group.length === 1) return group[0];
    const series = group.flatMap((source) => source.chart.series.map((item) => ({
      ...item,
      name: source.chart.series.length === 1 ? metricName(source) : `${metricName(source)}: ${item.name}`,
    })));
    if (series.length > 6) return group[0];
    const period = group[0].title.split(" · ").at(-1);
    return {
      id: group.map((source) => source.id).join(" · "),
      title: `${group.map(metricName).join(" vs. ")} · ${period}`,
      chart: { ...group[0].chart, series },
    };
  });
}
