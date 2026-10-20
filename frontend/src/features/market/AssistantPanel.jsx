import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { streamAssistant } from "./api";
import Icon from "../../shared/Icon";

const STORAGE_KEY = "enercol_market_chat_v1";
const STARTERS = [
  "Compara el precio de bolsa con el precio de escasez este mes.",
  "¿Cómo están los embalses frente a hace un año?",
  "¿Cuánto aportó la generación solar la última semana?",
  "¿Cuánto generó Hidroituango en los últimos 7 días?",
];

function loadMessages() {
  try {
    const stored = JSON.parse(window.localStorage.getItem(STORAGE_KEY));
    if (Array.isArray(stored)) return stored;
  } catch { /* Storage can be unavailable; start empty. */ }
  return [];
}

export default function AssistantPanel({ draft, onDraftUsed, describeCanvas, onCharts, onShowChart, open, onClose }) {
  const [messages, setMessages] = useState(loadMessages);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("");
  const [sending, setSending] = useState(false);
  const [focusChartId, setFocusChartId] = useState(null);
  const inputRef = useRef(null);
  const endRef = useRef(null);

  useEffect(() => {
    try { window.localStorage.setItem(STORAGE_KEY, JSON.stringify(messages.slice(-20))); } catch { /* Not persisted. */ }
    endRef.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  useEffect(() => {
    if (!draft) return;
    setInput(draft.text);
    setFocusChartId(draft.chartId);
    inputRef.current?.focus();
    onDraftUsed();
  }, [draft, onDraftUsed]);

  async function send(question) {
    const text = question.trim();
    if (!text || sending) return;
    const history = messages.filter((message) => !message.error).slice(-12).map(({ role, content }) => ({ role, content: content.slice(0, 6000) }));
    const assistantId = `${Date.now()}`;
    setMessages((current) => [...current, { id: `${assistantId}-q`, role: "user", content: text }, { id: assistantId, role: "assistant", content: "", charts: [] }]);
    setInput("");
    setFocusChartId(null);
    setSending(true);
    setStatus("Conectando");
    const patch = (changes) => setMessages((current) => current.map((message) => (message.id === assistantId ? { ...message, ...changes } : message)));
    try {
      let answer = "";
      await streamAssistant({ question: text, history, canvas: describeCanvas(focusChartId) }, (event) => {
        if (event.error) throw new Error(event.error);
        if (event.activity) setStatus(event.activity.status === "complete" ? event.activity.summary : "Consultando XM");
        if (event.delta) {
          answer += event.delta;
          patch({ content: answer });
        }
        if (event.answer) {
          answer = event.answer;
          patch({ content: answer });
        }
        if (event.audit) patch({ charts: onCharts(event.audit.sources) });
      });
      if (!answer) throw new Error("El asistente no produjo una respuesta. Inténtalo de nuevo.");
      setStatus("");
    } catch (error) {
      const message = error instanceof TypeError ? "No se pudo conectar con EnerCol." : error.message;
      patch({ content: `No se pudo completar la consulta. ${message}`, error: true });
      setStatus("");
    } finally {
      setSending(false);
    }
  }

  return (
    <aside className={`market-assistant ${open ? "is-open" : ""}`} aria-label="Asistente de mercado">
      <header className="market-assistant-head">
        <div>
          <div className="sidebar-label">Asistente</div>
          <strong>Pregunta y grafica</strong>
        </div>
        <div className="market-assistant-tools">
          {messages.length > 0 && <button type="button" className="market-card-button is-quiet" onClick={() => setMessages([])}>Limpiar</button>}
          <button type="button" className="market-card-button is-quiet market-assistant-close" aria-label="Ocultar asistente" title="Ocultar asistente" onClick={onClose}>×</button>
        </div>
      </header>
      <div className="market-assistant-thread">
        {!messages.length && <div className="market-assistant-empty">
          <p>Pregunta por precios, demanda, generación o embalses. Las series que consulte el asistente aparecen como gráficas en el lienzo, y puede leer las que ya tienes.</p>
          {STARTERS.map((starter) => <button key={starter} type="button" onClick={() => send(starter)}>{starter}</button>)}
        </div>}
        {messages.map((message) => <div key={message.id} className={`market-message is-${message.role} ${message.error ? "is-error" : ""}`}>
          {message.role === "assistant" && !message.content
            ? <div className="thinking-indicator"><span /><span /><span /> {status || "Pensando"}</div>
            : <div className="markdown-body"><ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown></div>}
          {message.charts?.length > 0 && <div className="market-message-charts">
            {message.charts.map((chart) => <button key={chart.id} type="button" onClick={() => onShowChart(chart.id)}>
              <Icon name="chart" size={14} /> Gráfica añadida al lienzo
            </button>)}
          </div>}
        </div>)}
        <div ref={endRef} />
      </div>
      <form className="market-assistant-composer" onSubmit={(event) => { event.preventDefault(); send(input); }}>
        <textarea
          ref={inputRef}
          value={input}
          rows={3}
          aria-label="Pregunta al asistente de mercado"
          placeholder="Ej. ¿Cómo se comportó la demanda en septiembre frente a agosto?"
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); send(input); } }}
        />
        <button type="submit" className="primary-action" disabled={sending || input.trim().length < 3}>{sending ? "Consultando…" : "Enviar"}</button>
      </form>
    </aside>
  );
}
