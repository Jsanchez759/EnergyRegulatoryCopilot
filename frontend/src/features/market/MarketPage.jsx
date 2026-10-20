import { useCallback, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import AssistantPanel from "./AssistantPanel";
import { seriesQueryOptions } from "./api";
import { chartsFromSources, describeChart, mergeLayers, rangeQuery } from "./canvas";
import CanvasCard from "./CanvasCard";
import ChartBuilder from "./ChartBuilder";
import Overview from "./Overview";
import useCanvas, { makeChartId } from "./useCanvas";
import Icon from "../../shared/Icon";

const MAX_CONTEXT_CHARTS = 4;

export default function MarketPage() {
  const canvas = useCanvas();
  const queryClient = useQueryClient();
  const [draft, setDraft] = useState(null);
  const [highlighted, setHighlighted] = useState(new Set());
  const [assistantOpen, setAssistantOpen] = useState(false);

  const showChart = useCallback((chartId) => {
    document.getElementById(`grafica-${chartId}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, []);

  const { addCharts: addToCanvas } = canvas;
  const addCharts = useCallback((created) => {
    addToCanvas(created);
    setHighlighted(new Set(created.map((chart) => chart.id)));
    window.setTimeout(() => setHighlighted(new Set()), 2400);
    if (created[0]) window.setTimeout(() => showChart(created[0].id), 50);
  }, [addToCanvas, showChart]);

  // The assistant reads what is on the canvas from the query cache, not from the screen.
  const describeCanvas = useCallback((focusId) => {
    const ordered = [...canvas.charts].sort((a, b) => (b.id === focusId) - (a.id === focusId));
    return ordered.slice(0, MAX_CONTEXT_CHARTS).flatMap((chart) => {
      const merged = mergeLayers(chart.layers.map((layer) => queryClient.getQueryData(seriesQueryOptions(layer).queryKey)));
      return merged ? [`${chart.id === focusId ? "[Gráfica sobre la que pregunta] " : ""}${describeChart(merged)}`] : [];
    });
  }, [canvas.charts, queryClient]);

  const onAssistantCharts = useCallback((sources) => {
    const created = chartsFromSources(sources, makeChartId);
    if (created.length) addCharts(created);
    return created.map((chart) => ({ id: chart.id }));
  }, [addCharts]);

  return (
    <div className={`market-page ${assistantOpen ? "assistant-open" : ""}`}>
      <div className="market-main">
        <header className="corpus-hero compact">
          <div className="welcome-rule"><span /> MERCADO ELÉCTRICO</div>
          <h1>Explora el mercado mayorista <em>con datos de XM.</em></h1>
          <p>Precios, demanda, generación, embalses y más, en vivo desde Sinergox. Arma tus gráficas o pídeselas al asistente.</p>
        </header>
        <Overview onOpen={(indicator) => addCharts([{ id: makeChartId(), type: "line", baseUnit: undefined, layers: [rangeQuery(indicator.metric_id, 90, indicator.entity)] }])} />
        <ChartBuilder
          charts={canvas.charts}
          onCreate={(query, baseUnit, type) => addCharts([{ id: makeChartId(), type, baseUnit, layers: [query] }])}
          onAddLayer={(chartId, query) => { canvas.addLayer(chartId, query); showChart(chartId); }}
        />
        <section className="market-canvas" aria-label="Lienzo de gráficas">
          <div className="market-canvas-head">
            <h2>Lienzo</h2>
            <button type="button" className="market-card-button is-quiet" onClick={canvas.reset}>Restablecer</button>
          </div>
          {canvas.charts.length
            ? canvas.charts.map((chart, index) => <CanvasCard
              key={chart.id}
              chart={chart}
              index={index}
              highlighted={highlighted.has(chart.id)}
              onUpdate={canvas.update}
              onRemove={canvas.remove}
              onRemoveLayer={canvas.removeLayer}
              onAsk={(target, merged) => {
                setAssistantOpen(true);
                setDraft({ chartId: target.id, text: `¿Qué explica el comportamiento de «${merged.title}» en este periodo?` });
              }}
            />)
            : <p className="monitor-empty">El lienzo está vacío. Crea una gráfica o pídesela al asistente.</p>}
        </section>
      </div>
      <AssistantPanel
        draft={draft}
        onDraftUsed={() => setDraft(null)}
        describeCanvas={describeCanvas}
        onCharts={onAssistantCharts}
        onShowChart={showChart}
        open={assistantOpen}
        onClose={() => setAssistantOpen(false)}
      />
      <button type="button" className="market-assistant-fab" onClick={() => setAssistantOpen(true)}>
        <Icon name="spark" size={16} /> Asistente
      </button>
    </div>
  );
}
