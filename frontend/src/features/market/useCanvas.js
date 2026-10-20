import { useCallback, useEffect, useState } from "react";

import { defaultCanvas } from "./canvas";

const STORAGE_KEY = "enercol_market_canvas_v1";
const MAX_CHARTS = 12;

function load() {
  try {
    const stored = JSON.parse(window.localStorage.getItem(STORAGE_KEY));
    if (Array.isArray(stored)) return stored;
  } catch { /* Storage can be unavailable; start from the default canvas. */ }
  return defaultCanvas();
}

export function makeChartId() {
  return `g${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
}

export default function useCanvas() {
  const [charts, setCharts] = useState(load);

  useEffect(() => {
    try { window.localStorage.setItem(STORAGE_KEY, JSON.stringify(charts)); } catch { /* Not persisted. */ }
  }, [charts]);

  const addCharts = useCallback((created) => {
    setCharts((current) => [...created, ...current].slice(0, MAX_CHARTS));
  }, []);
  const addLayer = useCallback((chartId, layer) => {
    setCharts((current) => current.map((chart) => (chart.id === chartId ? { ...chart, layers: [...chart.layers, layer] } : chart)));
  }, []);
  const update = useCallback((chartId, changes) => {
    setCharts((current) => current.map((chart) => (chart.id === chartId ? { ...chart, ...changes } : chart)));
  }, []);
  const removeLayer = useCallback((chartId, index) => {
    setCharts((current) => current
      .map((chart) => (chart.id === chartId ? { ...chart, layers: chart.layers.filter((_, position) => position !== index) } : chart))
      .filter((chart) => chart.layers.length));
  }, []);
  const remove = useCallback((chartId) => setCharts((current) => current.filter((chart) => chart.id !== chartId)), []);
  const reset = useCallback(() => setCharts(defaultCanvas()), []);

  return { charts, addCharts, addLayer, update, removeLayer, remove, reset };
}
