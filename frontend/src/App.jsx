import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

function safeError(error) {
  return error?.message || "Error inesperado.";
}

function nowLabel() {
  return new Date().toLocaleTimeString("es-CO", { hour: "2-digit", minute: "2-digit" });
}

function buildAuthHeaders(apiKey) {
  return apiKey ? { "X-OpenRouter-Api-Key": apiKey } : {};
}

function createChat(index = 1) {
  return {
    id: crypto.randomUUID(),
    title: `Conversacion ${index}`,
    messages: [
      {
        id: crypto.randomUUID(),
        role: "assistant",
        mode: "ask",
        content:
          "Hola. Este es un chat especializado en regulacion energetica colombiana. Puedes usar modo Ask o modo RAG con documentos indexados.",
        meta: "Sistema",
        time: nowLabel(),
      },
    ],
  };
}

async function requestJson(url, options = {}, apiKey = "") {
  const response = await fetch(url, {
    ...options,
    headers: {
      ...(options.headers || {}),
      ...buildAuthHeaders(apiKey),
    },
  });
  const text = await response.text();
  const data = (() => {
    try {
      return JSON.parse(text);
    } catch {
      return { raw: text };
    }
  })();
  if (!response.ok) throw new Error(JSON.stringify({ status: response.status, data }, null, 2));
  return data;
}

async function consumeSseResponse(response, onEvent) {
  if (!response.body) throw new Error("La respuesta no incluye stream de datos.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";
    for (const rawEvent of events) {
      const line = rawEvent.split("\n").find((part) => part.startsWith("data: "));
      if (!line) continue;
      try {
        onEvent(JSON.parse(line.slice(6)));
      } catch {
        // Ignore malformed chunks.
      }
    }
  }
}

function historyForApi(messages) {
  return messages
    .filter(
      (item) =>
        (item.role === "user" || item.role === "assistant") &&
        typeof item.content === "string" &&
        item.content.trim().length > 0
    )
    .slice(-12)
    .map((item) => ({ role: item.role, content: item.content }));
}

export default function App() {
  const [openRouterApiKey, setOpenRouterApiKey] = useState("");
  const [showApiHint, setShowApiHint] = useState(false);
  const [theme, setTheme] = useState(localStorage.getItem("erc_theme") || "light");
  const [showLeftPanel, setShowLeftPanel] = useState(true);
  const [showDocsPanel, setShowDocsPanel] = useState(true);
  const [mode, setMode] = useState("ask");
  const [topK, setTopK] = useState(4);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [streamStatus, setStreamStatus] = useState("inactivo");
  const [pdfFile, setPdfFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState(false);
  const [sidebarStatus, setSidebarStatus] = useState("Configura la API key para empezar.");
  const [chats, setChats] = useState([createChat()]);
  const [activeChatId, setActiveChatId] = useState("");
  const listRef = useRef(null);
  const isApiKeyMissing = openRouterApiKey.trim().length === 0;

  useEffect(() => {
    if (!activeChatId && chats.length) setActiveChatId(chats[0].id);
  }, [activeChatId, chats]);

  useEffect(() => {
    document.body.dataset.theme = theme;
    localStorage.setItem("erc_theme", theme);
  }, [theme]);

  useEffect(() => {
    if (isApiKeyMissing) return;
    void loadDocuments();
  }, [isApiKeyMissing]);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [chats, activeChatId, isSending]);

  const activeChat = useMemo(() => chats.find((chat) => chat.id === activeChatId) || chats[0], [chats, activeChatId]);
  const messages = activeChat?.messages || [];

  function updateActiveChat(updater) {
    setChats((prev) =>
      prev.map((chat) => (chat.id === activeChatId ? { ...chat, messages: updater(chat.messages) } : chat))
    );
  }

  function createNewChat() {
    const newChat = createChat(chats.length + 1);
    setChats((prev) => [newChat, ...prev]);
    setActiveChatId(newChat.id);
  }

  function deleteChat(chatId) {
    if (chats.length === 1) {
      const fresh = createChat(1);
      setChats([fresh]);
      setActiveChatId(fresh.id);
      return;
    }
    const next = chats.filter((chat) => chat.id !== chatId);
    setChats(next);
    if (activeChatId === chatId && next.length) setActiveChatId(next[0].id);
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // Ignore clipboard errors.
    }
  }

  async function loadDocuments() {
    setIsLoadingDocs(true);
    setSidebarStatus("Cargando documentos...");
    try {
      const data = await requestJson(`${API_BASE_URL}/rag/documents`, {}, openRouterApiKey);
      const docs = data.documents || [];
      setDocuments(docs);
      setSidebarStatus(docs.length ? "" : "No hay documentos indexados todavia.");
    } catch (error) {
      setSidebarStatus(`No se pudo cargar el listado: ${safeError(error)}`);
    } finally {
      setIsLoadingDocs(false);
    }
  }

  async function onUploadDocument(event) {
    event.preventDefault();
    if (isApiKeyMissing) {
      setShowApiHint(true);
      setSidebarStatus("Debes ingresar tu OPENROUTER_API_KEY para continuar.");
      return;
    }
    if (!pdfFile) {
      setSidebarStatus("Selecciona un PDF antes de subir.");
      return;
    }

    setIsUploading(true);
    setSidebarStatus("Subiendo e indexando documento...");
    try {
      const formData = new FormData();
      formData.append("file", pdfFile);
      const result = await requestJson(`${API_BASE_URL}/rag/documents/upload`, { method: "POST", body: formData }, openRouterApiKey);
      setPdfFile(null);
      setSidebarStatus(`Documento indexado: ${result.filename} (${result.chunks_indexed} chunks).`);
      await loadDocuments();
    } catch (error) {
      setSidebarStatus(`Error de carga: ${safeError(error)}`);
    } finally {
      setIsUploading(false);
    }
  }

  async function onSendMessage(event) {
    event.preventDefault();
    const question = input.trim();
    if (isApiKeyMissing) {
      setShowApiHint(true);
      setStreamStatus("API key requerida");
      return;
    }
    if (!question || isSending || !activeChatId) return;

    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      mode,
      content: question,
      meta: mode === "rag" ? `RAG (top_k=${topK})` : "Ask",
      time: nowLabel(),
    };

    updateActiveChat((prev) => [...prev, userMessage]);
    setInput("");
    setIsSending(true);
    setStreamStatus("conectando");

    try {
      const assistantId = crypto.randomUUID();
      updateActiveChat((prev) => [
        ...prev,
        {
          id: assistantId,
          role: "assistant",
          mode,
          content: "",
          references: [],
          meta: mode === "rag" ? "Respuesta RAG" : "Respuesta Ask",
          time: nowLabel(),
        },
      ]);

      const target = mode === "rag" ? `${API_BASE_URL}/rag/ask/stream` : `${API_BASE_URL}/questions/ask/stream`;
      const body =
        mode === "rag"
          ? { question, top_k: topK }
          : { question, history: historyForApi([...messages, userMessage]) };

      const response = await fetch(target, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...buildAuthHeaders(openRouterApiKey),
        },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `Error HTTP ${response.status}`);
      }
      setStreamStatus("recibiendo");

      await consumeSseResponse(response, (evt) => {
        if (evt.error) throw new Error(evt.error);
        if (evt.references) {
          updateActiveChat((prev) =>
            prev.map((item) => (item.id === assistantId ? { ...item, references: evt.references } : item))
          );
        }
        if (evt.delta) {
          updateActiveChat((prev) =>
            prev.map((item) =>
              item.id === assistantId ? { ...item, content: `${item.content}${evt.delta}` } : item
            )
          );
        }
        if (evt.done) {
          setStreamStatus("finalizado");
        }
      });
    } catch (error) {
      updateActiveChat((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          mode,
          content: `Error al consultar el backend:\n${safeError(error)}`,
          meta: "Error",
          time: nowLabel(),
        },
      ]);
      setStreamStatus("error");
    } finally {
      setIsSending(false);
    }
  }

  return (
    <div className="app">
      <div className="bg-layer" />
      <main
        className={`shell three-cols ${showLeftPanel ? "" : "hide-left"} ${showDocsPanel ? "" : "hide-right"}`}
        style={{ display: "grid" }}
      >
        {showLeftPanel ? (
          <aside className="left-pane">
          <div className="left-head">
            <h2>Conversaciones</h2>
            <button type="button" className="ghost mini" onClick={() => setShowLeftPanel(false)}>
              Ocultar
            </button>
          </div>
          <button type="button" className="ghost" onClick={createNewChat}>
            Nueva conversacion
          </button>
          <div className="chat-list">
            {chats.map((chat) => (
              <div key={chat.id} className={`chat-item ${chat.id === activeChatId ? "active" : ""}`}>
                <button type="button" className="chat-select" onClick={() => setActiveChatId(chat.id)}>
                  {chat.title}
                </button>
                <button type="button" className="chat-delete" onClick={() => deleteChat(chat.id)}>
                  Borrar
                </button>
              </div>
            ))}
          </div>
          <div className="guide-box">
            <h3>Como usar</h3>
            <p>1. Selecciona modo Ask para preguntas generales o RAG para usar documentos.</p>
            <p>2. En modo RAG, sube PDF y luego consulta.</p>
            <p>3. Revisa referencias expandibles para ver chunks fuente.</p>
          </div>
          </aside>
        ) : null}

        <section className="chat-pane">
          <header className="chat-header">
            <div>
              <p className="kicker">EnerCol Chat</p>
              <h1>Consultor IA de Normativa CREG y Mercado Electrico</h1>
            </div>
            <div className="mode-switch">
              {!showLeftPanel ? (
                <button type="button" onClick={() => setShowLeftPanel(true)}>
                  Mostrar panel
                </button>
              ) : null}
              {!showDocsPanel ? (
                <button type="button" onClick={() => setShowDocsPanel(true)}>
                  Mostrar documentos
                </button>
              ) : null}
              <button type="button" className={mode === "ask" ? "active" : ""} onClick={() => setMode("ask")}>
                Modo Ask
              </button>
              <button type="button" className={mode === "rag" ? "active" : ""} onClick={() => setMode("rag")}>
                Modo RAG
              </button>
            </div>
          </header>

          <div className="connection-row">
            <label>API Key</label>
            <div className="key-badge">
              <input
                type="password"
                value={openRouterApiKey}
                onChange={(e) => setOpenRouterApiKey(e.target.value)}
                placeholder="sk-or-v1-..."
              />
              <button
                type="button"
                className="ghost mini"
                onClick={() => setShowApiHint((v) => !v)}
              >
                ?
              </button>
              {showApiHint ? (
                <div className="api-popover">
                  <p>
                    Debes ingresar tu <code>OPENROUTER_API_KEY</code> para usar el chat.
                  </p>
                  <p>
                    Esta app usa modelos <strong>free</strong> de OpenRouter.
                  </p>
                  <p className="aside-note">
                    <a href="https://www.youtube.com/watch?v=ZELx_OzYAQo" target="_blank" rel="noreferrer">
                      Video paso a paso
                    </a>
                    {" · "}
                    <a href="https://openrouter.ai/" target="_blank" rel="noreferrer">
                      Portal oficial
                    </a>
                  </p>
                </div>
              ) : null}
            </div>
            <label>Tema</label>
            <button type="button" className="ghost mini" onClick={() => setTheme((t) => (t === "light" ? "dark" : "light"))}>
              {theme === "light" ? "Oscuro" : "Claro"}
            </button>
            {mode === "rag" ? (
              <>
                <label>Top K</label>
                <input type="number" min={1} max={10} value={topK} onChange={(e) => setTopK(Number(e.target.value) || 4)} />
              </>
            ) : null}
          </div>

          <div className="stream-badge">Estado streaming: {streamStatus}</div>

          <div className="message-list" ref={listRef}>
            {messages.map((message) => (
              <article key={message.id} className={`bubble ${message.role}`}>
                <div className="meta-row">
                  <p className="meta">{message.meta}</p>
                  <span className="time">{message.time}</span>
                </div>
                <div className="content markdown-body">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                </div>
                {message.role === "assistant" && message.references?.length ? (
                  <div className="refs-box">
                    <p className="refs-title">Referencias</p>
                    {message.references.map((ref) => (
                      <details key={ref.id} className="ref-item">
                        <summary>
                          {ref.label}
                          {typeof ref.distance === "number" ? <span className="distance">score: {ref.distance.toFixed(4)}</span> : null}
                        </summary>
                        <div className="ref-chunk markdown-body">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{ref.chunk}</ReactMarkdown>
                          <button type="button" className="ghost mini" onClick={() => void copyText(ref.chunk)}>
                            Copiar chunk
                          </button>
                        </div>
                      </details>
                    ))}
                  </div>
                ) : null}
              </article>
            ))}
            {isSending ? (
              <article className="bubble assistant pending">
                <p className="meta">Asistente</p>
                <div className="typing" aria-label="Generando respuesta">
                  <span />
                  <span />
                  <span />
                </div>
              </article>
            ) : null}
          </div>

          <form className="composer" onSubmit={onSendMessage}>
            <textarea
              rows={3}
              placeholder={mode === "rag" ? "Pregunta sobre documentos indexados..." : "Pregunta general sobre regulacion energetica..."}
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
            <button type="submit" disabled={isSending}>
              {isSending ? (
                <span className="btn-inline">
                  <span className="spinner" />
                  Enviando...
                </span>
              ) : (
                "Enviar"
              )}
            </button>
          </form>
        </section>

        {showDocsPanel ? (
          <aside className="docs-pane">
          <div className="left-head">
            <h2>Documentos Indexados</h2>
            <button type="button" className="ghost mini" onClick={() => setShowDocsPanel(false)}>
              Ocultar
            </button>
          </div>
          <p className="aside-note">Sube documentos para consultas RAG con evidencia.</p>

          <form className="upload-form" onSubmit={onUploadDocument}>
            <input type="file" accept="application/pdf" onChange={(e) => setPdfFile(e.target.files?.[0] || null)} />
            <button type="submit" disabled={isUploading}>
              {isUploading ? (
                <span className="btn-inline">
                  <span className="spinner" />
                  Procesando...
                </span>
              ) : (
                "Subir PDF"
              )}
            </button>
          </form>

          <button className="ghost" type="button" onClick={() => void loadDocuments()}>
            Actualizar listado
          </button>

          {sidebarStatus ? <p className="status">{sidebarStatus}</p> : null}

          {isLoadingDocs ? (
            <div className="doc-skeletons">
              <div className="skeleton-item" />
              <div className="skeleton-item" />
              <div className="skeleton-item" />
            </div>
          ) : (
            <div className="doc-list">
              {documents.map((doc) => (
                <article key={doc.document_id} className="doc-item">
                  <p className="doc-name">{doc.filename}</p>
                  <p className="doc-meta">{doc.chunks_indexed} chunks indexados</p>
                </article>
              ))}
            </div>
          )}
          </aside>
        ) : null}
      </main>
    </div>
  );
}
