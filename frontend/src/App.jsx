import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import AccountPanel from "./features/auth/AccountPanel";
import AuthScreen from "./features/auth/AuthScreen";
import { authApi, authQueryKey } from "./features/auth/api";
import {
  conversationsQueryKey,
  useConversationMessages,
  useConversations,
  useCreateConversation,
  useDeleteConversation,
} from "./features/chat/api";
import { consumeSseResponse } from "./features/chat/stream";
import { useDocuments } from "./features/documents/queries";
import SourceLibrary from "./features/documents/SourceLibrary";
import { API_BASE_URL } from "./lib/api";
import Icon from "./shared/Icon";
import useResizableWidth from "./shared/useResizableWidth";

const THEME_STORAGE_KEY = "enercol_theme_v1";
const AUTH_SYNC_KEY = "enercol_auth_changed";
const HISTORY_WIDTH_STORAGE_KEY = "enercol_history_width_v1";
const NO_MESSAGES = [];
const STARTERS = [
  "¿Qué establece la regulación sobre autogeneración?",
  "Resume las reglas para conectar un proyecto solar.",
  "¿Qué obligaciones tiene un comercializador de energía?",
];

function nowLabel(value) {
  const date = value ? new Date(value) : new Date();
  return date.toLocaleTimeString("es-CO", { hour: "2-digit", minute: "2-digit" });
}

function makeId() {
  return crypto.randomUUID();
}

function AppLoading() {
  return <main className="app-loading"><span className="brand-mark"><span /><span /><span /></span><span>Preparando EnerCol…</span></main>;
}

function AppUnavailable({ onRetry }) {
  return <main className="app-unavailable"><span className="brand-mark"><span /><span /><span /></span><h1>No pudimos conectar.</h1><p>Comprueba que el servidor esté activo e inténtalo de nuevo.</p><button type="button" onClick={onRetry}>Reintentar</button></main>;
}

function AuditPanel({ audit }) {
  if (!audit) return null;
  return (
    <details className="audit-panel">
      <summary><Icon name="file" size={15} /><span>Auditoría · {audit.sources.length} fuentes</span><span className="reference-chevron">⌄</span></summary>
      <div className="audit-content">
        <p className="audit-method">{audit.method}</p>
        {audit.steps.length > 0 && <ol className="audit-steps">
          {audit.steps.map((step, index) => <li key={`${step.tool}-${index}`}>
            <strong>{step.tool === "search_uploaded_documents" ? "Documentos" : "Web"}</strong>
            <span>{step.summary}</span><code>{step.query}</code>
          </li>)}
        </ol>}
        {audit.sources.length > 0 ? <ul className="audit-sources">
          {audit.sources.map((source) => <li key={source.id}>
            <span className="source-id">{source.id}</span>
            <div className="audit-source-copy">
              <strong>{source.url ? <a href={source.url} target="_blank" rel="noreferrer">{source.title}</a> : source.title}</strong>
              {source.page && <small>Página {source.page}</small>}
              {source.file_sha256 && <small>SHA-256 · {source.file_sha256}</small>}
              <details><summary>Ver evidencia</summary><p>{source.snippet}</p></details>
            </div>
          </li>)}
        </ul> : <p className="audit-empty">No se encontraron fuentes verificables.</p>}
      </div>
    </details>
  );
}

function AuthenticatedWorkspace({ user }) {
  const queryClient = useQueryClient();
  const [activeConversationId, setActiveConversationId] = useState("");
  const [theme, setTheme] = useState(() => localStorage.getItem(THEME_STORAGE_KEY) || "light");
  const [input, setInput] = useState("");
  const [conversationSearch, setConversationSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [streamStatus, setStreamStatus] = useState("Listo");
  const [showHistory, setShowHistory] = useState(() => !window.matchMedia("(max-width: 700px)").matches);
  const [showSources, setShowSources] = useState(() => !window.matchMedia("(max-width: 1020px)").matches);
  const [showAccount, setShowAccount] = useState(false);
  const [appError, setAppError] = useState("");
  const historyResize = useResizableWidth({
    storageKey: HISTORY_WIDTH_STORAGE_KEY,
    defaultWidth: 260,
    minWidth: 200,
    maxWidth: 480,
  });
  const messagesEndRef = useRef(null);
  const searchInputRef = useRef(null);
  const initialConversationStarted = useRef(false);
  const allConversationsQuery = useConversations();
  const searchConversationsQuery = useConversations(debouncedSearch);
  const conversationsQuery = debouncedSearch ? searchConversationsQuery : allConversationsQuery;
  const messagesQuery = useConversationMessages(activeConversationId);
  const documentsQuery = useDocuments();
  const createConversation = useCreateConversation();
  const deleteConversation = useDeleteConversation();
  const conversations = conversationsQuery.data?.conversations || [];
  const allConversations = allConversationsQuery.data?.conversations || [];
  const activeConversation = allConversations.find((item) => item.id === activeConversationId) || allConversations[0];
  const messages = messagesQuery.data?.messages ?? NO_MESSAGES;
  const logoutMutation = useMutation({
    mutationFn: authApi.logout,
    onSuccess: () => {
      localStorage.setItem(AUTH_SYNC_KEY, String(Date.now()));
      queryClient.setQueryData(authQueryKey, null);
      queryClient.removeQueries({ queryKey: conversationsQueryKey });
      queryClient.removeQueries({ queryKey: ["documents"] });
    },
  });

  useEffect(() => {
    document.body.dataset.theme = theme;
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  useEffect(() => {
    const timeout = window.setTimeout(() => setDebouncedSearch(conversationSearch.trim()), 180);
    return () => window.clearTimeout(timeout);
  }, [conversationSearch]);

  useEffect(() => {
    function onSearchShortcut(event) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setShowHistory(true);
        window.setTimeout(() => searchInputRef.current?.focus(), 0);
      }
      if (event.key === "Escape" && document.activeElement === searchInputRef.current) {
        setConversationSearch("");
        searchInputRef.current?.blur();
      }
    }
    window.addEventListener("keydown", onSearchShortcut);
    return () => window.removeEventListener("keydown", onSearchShortcut);
  }, []);

  useEffect(() => {
    if (activeConversation && activeConversation.id !== activeConversationId) {
      setActiveConversationId(activeConversation.id);
    }
  }, [activeConversation, activeConversationId]);

  useEffect(() => {
    if (debouncedSearch || !allConversationsQuery.isSuccess || allConversations.length || initialConversationStarted.current) return;
    initialConversationStarted.current = true;
    createConversation.mutate(undefined, {
      onSuccess: (conversation) => setActiveConversationId(conversation.id),
    });
  }, [allConversationsQuery.isSuccess, allConversations.length, createConversation, debouncedSearch]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isSending]);

  async function createNewChat() {
    if (isSending || createConversation.isPending) return;
    setAppError("");
    try {
      const conversation = await createConversation.mutateAsync();
      setActiveConversationId(conversation.id);
      setStreamStatus("Listo");
    } catch (error) {
      setAppError(error.message);
    }
  }

  async function removeConversation(conversationId) {
    try {
      await deleteConversation.mutateAsync(conversationId);
      if (activeConversationId === conversationId) {
        const nextConversation = allConversations.find((item) => item.id !== conversationId);
        setActiveConversationId(nextConversation?.id || "");
      }
    } catch (error) {
      setAppError(error.message);
    }
  }

  async function onSendMessage(event) {
    event.preventDefault();
    const question = input.trim();
    const conversationId = activeConversation?.id;
    if (!question || isSending || !conversationId) return;

    const queryKey = [...conversationsQueryKey, conversationId, "messages"];
    const assistantId = makeId();
    const userMessage = { id: makeId(), role: "user", content: question, created_at: new Date().toISOString() };
    queryClient.setQueryData(queryKey, (current) => ({
      messages: [...(current?.messages || []), userMessage, {
        id: assistantId, role: "assistant", content: "", audit: null, created_at: new Date().toISOString(),
      }],
    }));
    setInput("");
    setIsSending(true);
    setAppError("");
    setStreamStatus("Conectando");
    let receivedText = false;

    try {
      const response = await fetch(`${API_BASE_URL}/agent/stream`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, conversation_id: conversationId }),
      });
      if (!response.ok) {
        let errorMessage = `Error del servidor (${response.status}).`;
        try { errorMessage = (await response.json()).detail || errorMessage; } catch { /* Keep the status message. */ }
        throw new Error(errorMessage);
      }
      setStreamStatus("En respuesta");
      await consumeSseResponse(response, (eventData) => {
        if (eventData.error) throw new Error(eventData.error);
        if (eventData.activity) setStreamStatus(eventData.activity.summary || "Consultando fuentes");
        if (eventData.audit) {
          queryClient.setQueryData(queryKey, (current) => ({
            ...current,
            messages: (current?.messages || []).map((message) => message.id === assistantId ? { ...message, audit: eventData.audit } : message),
          }));
        }
        if (eventData.delta) {
          receivedText = true;
          queryClient.setQueryData(queryKey, (current) => ({
            ...current,
            messages: (current?.messages || []).map((message) => message.id === assistantId ? { ...message, content: message.content + eventData.delta } : message),
          }));
        }
        if (eventData.answer) {
          receivedText = true;
          queryClient.setQueryData(queryKey, (current) => ({
            ...current,
            messages: (current?.messages || []).map((message) => message.id === assistantId ? { ...message, content: eventData.answer } : message),
          }));
        }
        if (eventData.done) setStreamStatus("Completado");
      });
      if (!receivedText) throw new Error("El agente finalizó sin producir una respuesta. Inténtalo de nuevo.");
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: conversationsQueryKey }),
        queryClient.invalidateQueries({ queryKey }),
      ]);
    } catch (error) {
      const userFacingError = error instanceof TypeError
        ? "No se pudo conectar con EnerCol. Comprueba que el servidor esté activo e inténtalo de nuevo."
        : error.message;
      setAppError("");
      queryClient.setQueryData(queryKey, (current) => ({
        messages: (current?.messages || []).map((message) => message.id === assistantId
          ? { ...message, content: `No se pudo completar la consulta. ${userFacingError}`, meta: "Error" }
          : message),
      }));
      setStreamStatus("Error");
    } finally {
      setIsSending(false);
    }
  }

  function exportConversation() {
    if (!activeConversation || !messages.length) return;
    const transcript = [
      `# ${activeConversation.title}`,
      "",
      `Exportado desde EnerCol el ${new Date().toLocaleString("es-CO")}`,
      "",
      ...messages.flatMap((message) => {
        const heading = message.role === "user" ? "## Consulta" : "## EnerCol";
        const audit = message.audit ? [
          "",
          `Método: ${message.audit.method}`,
          ...message.audit.sources.map((source) => {
            const detail = source.page ? `, página ${source.page}` : "";
            return `- [${source.id}] ${source.title}${detail}${source.url ? ` — ${source.url}` : ""}`;
          }),
        ] : [];
        return [heading, "", message.content, ...audit, ""];
      }),
    ].join("\n");
    const blobUrl = URL.createObjectURL(new Blob([transcript], { type: "text/markdown;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = blobUrl;
    const fileName = activeConversation.title.normalize("NFKD").replace(/[\u0300-\u036f]/g, "");
    link.download = `${fileName.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "consulta"}.md`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
  }

  function onDeleteAccount() {
    localStorage.setItem(AUTH_SYNC_KEY, String(Date.now()));
    queryClient.setQueryData(authQueryKey, null);
    queryClient.removeQueries({ queryKey: conversationsQueryKey });
    queryClient.removeQueries({ queryKey: ["documents"] });
    setShowAccount(false);
  }

  return (
    <main
      className={`workspace ${!showHistory ? "history-hidden" : ""} ${!showSources ? "sources-hidden" : ""} ${historyResize.isResizing ? "is-resizing" : ""}`}
      style={{ "--history-width": `${historyResize.width}px` }}
    >
      {showHistory && <aside className="history-pane" aria-label="Conversaciones">
        <a className="brand" href="#inicio" aria-label="EnerCol inicio">
          <span className="brand-mark"><span /><span /><span /></span><span>enercol<span className="brand-period">.</span></span>
        </a>
        <button className="mobile-close" type="button" onClick={() => setShowHistory(false)} aria-label="Cerrar conversaciones"><Icon name="close" size={18} /></button>
        <button className="new-chat-button" type="button" onClick={createNewChat} disabled={isSending}>
          <Icon name="plus" size={16} /> Nueva consulta
        </button>
        <div className="conversation-search">
          <Icon name="search" size={16} />
          <input
            ref={searchInputRef}
            type="search"
            value={conversationSearch}
            onChange={(event) => setConversationSearch(event.target.value)}
            placeholder="Buscar consultas"
            aria-label="Buscar en tus consultas"
          />
          {!conversationSearch && <kbd>⌘ K</kbd>}
          {conversationSearch && <button type="button" onClick={() => setConversationSearch("")} aria-label="Limpiar búsqueda">×</button>}
        </div>
        <div className="sidebar-label">{debouncedSearch ? "Resultados" : "Recientes"}</div>
        <nav className="conversation-list">
          {conversations.map((conversation) => <div key={conversation.id} className={`conversation-row ${conversation.id === activeConversation?.id ? "is-active" : ""}`}>
            <button type="button" className="conversation-select" onClick={() => setActiveConversationId(conversation.id)} title={conversation.title}>
              <span className="conversation-dot" /><span className="conversation-title">{conversation.title}</span>
            </button>
            <button className="conversation-delete" type="button" onClick={() => void removeConversation(conversation.id)} aria-label={`Eliminar ${conversation.title}`} title="Eliminar">×</button>
          </div>)}
          {debouncedSearch && conversations.length === 0 && <p className="conversation-empty">No hay consultas que coincidan.</p>}
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-caption">Asistente de investigación</div>
          <div className="sidebar-subcaption">Regulación energética · Colombia</div>
          <button className="theme-toggle" type="button" onClick={() => setTheme((value) => value === "light" ? "dark" : "light")}>
            <Icon name="sun" size={16} /> Apariencia <span>{theme === "light" ? "Claro" : "Oscuro"}</span>
          </button>
        </div>
        <div className="history-resize-handle" aria-label="Ajustar ancho del panel de conversaciones" title="Arrastra para ajustar · doble clic para restablecer" {...historyResize.handleProps} />
      </aside>}

      <section className="research-pane">
        <header className="topbar">
          <div className="breadcrumb"><span>Colombia</span><span className="breadcrumb-separator">/</span><strong>Investigación regulatoria</strong></div>
          <div className="topbar-actions">
            {!showHistory && <button className="icon-button" type="button" onClick={() => setShowHistory(true)} aria-label="Mostrar conversaciones" title="Conversaciones">☰</button>}
            <span className={`connection-indicator ${documentsQuery.isError ? "is-offline" : ""}`} />
            <span className="connection-label">{documentsQuery.isError ? "Backend sin conexión" : documentsQuery.isLoading ? "Conectando…" : "Espacio privado"}</span>
            <button className={`source-toggle ${!showSources ? "is-visible" : ""}`} type="button" onClick={() => setShowSources((value) => !value)}>{showSources ? "Ocultar fuentes" : "Ver fuentes"}</button>
            {messages.length > 0 && <button className="export-button" type="button" onClick={exportConversation} title="Descargar conversación con referencias">Exportar</button>}
            <button className="account-trigger" type="button" onClick={() => setShowAccount(true)} aria-label="Gestionar cuenta" title="Gestionar cuenta"><span>{user.display_name.slice(0, 1).toUpperCase()}</span></button>
            <button className="topbar-signout" type="button" onClick={() => logoutMutation.mutate()} disabled={logoutMutation.isPending}>Salir</button>
          </div>
        </header>

        <div className="conversation-workspace">
          <div className="thread-toolbar">
            <span className="agent-label">AGENTE DE INVESTIGACIÓN</span>
            <span className="stream-status"><span className={isSending ? "status-pulse" : "status-dot"} />{streamStatus}</span>
          </div>
          <div className="message-scroll">
            {appError && <p className="inline-app-error" role="status">{appError}</p>}
            {messages.length === 0 ? <div className="welcome-view">
              <div className="welcome-rule"><span /> REGULACIÓN ENERGÉTICA DE COLOMBIA</div>
              <h1>¿Qué necesitas<br /><em>entender hoy?</em></h1>
              <p className="welcome-copy">Consulta regulación energética o busca dentro de tus propios documentos.</p>
              <div className="suggestion-list" aria-label="Preguntas sugeridas">
                {STARTERS.map((question, index) => <button key={question} type="button" onClick={() => setInput(question)}>
                  <span className="suggestion-index">0{index + 1}</span><span>{question}</span><span className="suggestion-arrow">↗</span>
                </button>)}
              </div>
              <div className="welcome-footnote"><span className="footnote-line" />Las respuestas generales pueden requerir verificación en fuentes oficiales.</div>
            </div> : <div className="message-thread">
              {messages.map((message) => <article key={message.id} className={`message-row ${message.role}`}>
                {message.role === "assistant" && <div className="assistant-mark"><span /><span /><span /></div>}
                <div className="message-column">
                  <div className="message-byline"><strong>{message.role === "user" ? "Tú" : message.meta || "EnerCol"}</strong><span>{nowLabel(message.created_at)}</span></div>
                  <div className="message-content markdown-body"><ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown></div>
                  {message.role === "assistant" && <AuditPanel audit={message.audit} />}
                </div>
              </article>)}
              {isSending && <div className="thinking-indicator"><span /><span /><span /> Preparando respuesta</div>}
              <div ref={messagesEndRef} />
            </div>}
          </div>
          <div className="composer-wrap">
            <form className="composer" onSubmit={onSendMessage}>
              <textarea
                rows={2}
                aria-label="Escribe tu consulta"
                placeholder="Pregunta sobre regulación energética o tus fuentes…"
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); event.currentTarget.form.requestSubmit(); } }}
              />
              <div className="composer-footer">
                <span><kbd>↵</kbd> enviar <span className="shortcut-divider">·</span> <kbd>⇧ ↵</kbd> nueva línea</span>
                <button type="submit" disabled={isSending || !input.trim() || !activeConversation?.id} aria-label="Enviar consulta"><Icon name="send" size={16} /></button>
              </div>
            </form>
            <p className="composer-note">EnerCol puede equivocarse. Verifica siempre la norma citada.</p>
          </div>
        </div>
      </section>

      {showSources && <SourceLibrary onClose={() => setShowSources(false)} />}
      {showAccount && <AccountPanel user={user} onClose={() => setShowAccount(false)} onDeleted={onDeleteAccount} />}
    </main>
  );
}

export default function App() {
  const userQuery = useQuery({
    queryKey: authQueryKey,
    queryFn: authApi.me,
    retry: (failureCount, error) => error.status !== 401 && failureCount < 2,
  });
  const queryClient = useQueryClient();
  useEffect(() => {
    const onUnauthorized = () => {
      queryClient.setQueryData(authQueryKey, null);
      queryClient.removeQueries({ queryKey: conversationsQueryKey });
      queryClient.removeQueries({ queryKey: ["documents"] });
    };
    const onStorage = (event) => {
      if (event.key === AUTH_SYNC_KEY) void queryClient.invalidateQueries({ queryKey: authQueryKey });
    };
    window.addEventListener("enercol:unauthorized", onUnauthorized);
    window.addEventListener("storage", onStorage);
    return () => {
      window.removeEventListener("enercol:unauthorized", onUnauthorized);
      window.removeEventListener("storage", onStorage);
    };
  }, [queryClient]);
  if (userQuery.isLoading) return <AppLoading />;
  if (userQuery.data) return <AuthenticatedWorkspace user={userQuery.data} />;
  if (userQuery.error?.status === 401 || userQuery.data === null) return <AuthScreen />;
  return <AppUnavailable onRetry={() => userQuery.refetch()} />;
}
