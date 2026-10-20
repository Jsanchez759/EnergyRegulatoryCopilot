import { describe, expect, it } from "vitest";

import { chartsFromSources, defaultCanvas, describeChart, mergeLayers, toCsv } from "./canvas";

function result(name, x, values, unit = "COP/kWh") {
  return {
    metric: { name },
    start: "2026-09-01",
    end: "2026-09-03",
    chart: { unit, x, series: [{ name: "Sistema", values }] },
  };
}

describe("mergeLayers", () => {
  it("aligns layers on the union of periods and names them by metric", () => {
    const merged = mergeLayers([
      result("Precio Bolsa Nacional Ponderado por Sistema", ["2026-09-01", "2026-09-02"], [900, 800]),
      result("Precio Escasez por Sistema", ["2026-09-02", "2026-09-03"], [698, 698]),
    ]);
    expect(merged.chart.x).toEqual(["2026-09-01", "2026-09-02", "2026-09-03"]);
    expect(merged.chart.series).toEqual([
      { name: "Precio Bolsa Nacional Ponderado", values: [900, 800, null] },
      { name: "Precio Escasez", values: [null, 698, 698] },
    ]);
    expect(merged.title).toBe("Precio Bolsa Nacional Ponderado por Sistema vs. Precio Escasez por Sistema");
  });

  it("keeps breakdown names for a single layer and waits for data", () => {
    const technology = {
      ...result("Generación por Recurso", ["2026-09-01"], [1], "GWh"),
      chart: { unit: "GWh", x: ["2026-09-01"], series: [{ name: "Hidraulica", values: [140] }, { name: "Solar", values: [18] }] },
    };
    expect(mergeLayers([technology]).chart.series.map((item) => item.name)).toEqual(["Hidraulica", "Solar"]);
    expect(mergeLayers([undefined])).toBeNull();
  });
});

describe("describeChart and toCsv", () => {
  const merged = mergeLayers([result("Precio Escasez por Sistema", ["2026-09-01", "2026-09-02"], [690.5, 700])]);

  it("summarizes each series for the assistant", () => {
    const text = describeChart(merged);
    expect(text).toContain("Precio Escasez por Sistema (2026-09-01 a 2026-09-03, COP/kWh)");
    expect(text).toContain("máximo 700 (2026-09-02)");
  });

  it("exports one column per series", () => {
    expect(toCsv(merged).split("\n")).toEqual(["periodo,Precio Escasez", "2026-09-01,690.5", "2026-09-02,700"]);
  });
});

describe("chartsFromSources", () => {
  it("turns the assistant's comparable series into one chart with a layer each", () => {
    const query = (metric) => ({ metric_id: metric, entity: "Sistema", start: "2026-09-01", end: "2026-09-30", codes: [], group_by_technology: false });
    const source = (id, metric, unit) => ({ id, kind: "market", chart: { unit, x: ["a", "b"], series: [], query: query(metric) } });
    let next = 0;
    const charts = chartsFromSources([
      source("M1", "PrecBolsNaci", "COP/kWh"),
      source("M2", "PrecEsca", "COP/kWh"),
      source("M3", "PorcVoluUtilDiar", "%"),
      { id: "R1", kind: "regulation" },
    ], () => `c${(next += 1)}`);
    expect(charts.map((chart) => chart.layers.map((layer) => layer.metric_id))).toEqual([["PrecBolsNaci", "PrecEsca"], ["PorcVoluUtilDiar"]]);
  });
});

describe("defaultCanvas", () => {
  it("starts with prices and generation by technology", () => {
    const [prices, generation] = defaultCanvas(new Date(2026, 9, 20));
    expect(prices.layers.map((layer) => [layer.metric_id, layer.start, layer.end])).toEqual([
      ["PPPrecBolsNaci", "2026-09-20", "2026-10-20"],
      ["PrecEsca", "2026-09-20", "2026-10-20"],
    ]);
    expect(generation.layers[0].group_by_technology).toBe(true);
  });
});
