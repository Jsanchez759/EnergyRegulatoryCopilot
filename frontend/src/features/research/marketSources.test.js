import { describe, expect, it } from "vitest";

import { combineMarketSources } from "./marketSources";

function source(id, name, unit, x, values) {
  return {
    id,
    kind: "market",
    title: `XM · ${name} · 2026-09-01 a 2026-09-03`,
    chart: { unit, x, series: [{ name: "Sistema", values }] },
  };
}

const DAYS = ["2026-09-01", "2026-09-02", "2026-09-03"];

describe("combineMarketSources", () => {
  it("puts series with the same unit and dates on one chart named by metric", () => {
    const [combined] = combineMarketSources([
      source("M1", "Precio Bolsa Nacional por Sistema", "COP/kWh", DAYS, [900, 800, 700]),
      source("M2", "Precio Escasez por Sistema", "COP/kWh", DAYS, [698, 698, 698]),
    ]);
    expect(combined.id).toBe("M1 · M2");
    expect(combined.title).toBe("Precio Bolsa Nacional por Sistema vs. Precio Escasez por Sistema · 2026-09-01 a 2026-09-03");
    expect(combined.chart.series.map((series) => series.name)).toEqual([
      "Precio Bolsa Nacional por Sistema",
      "Precio Escasez por Sistema",
    ]);
  });

  it("keeps different units apart and ignores other sources", () => {
    const charts = combineMarketSources([
      source("M1", "Precio Bolsa", "COP/kWh", DAYS, [1, 2, 3]),
      source("M2", "Embalses", "%", DAYS, [70, 71, 72]),
      { id: "R1", kind: "regulation", title: "Resolución" },
    ]);
    expect(charts.map((chart) => chart.id)).toEqual(["M1", "M2"]);
  });
});
