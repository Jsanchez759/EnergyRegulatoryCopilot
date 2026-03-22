import asyncio
import threading

from openai import OpenAI

from app.core.config import settings
from app.schemas.questions import ChatHistoryItem, QuestionResponse


class QuestionService:
    """Service layer for question-answering logic."""

    def __init__(self) -> None:
        """Initialize optional OpenRouter chat client."""
        self._client = None
        if settings.OPENROUTER_API_KEY:
            self._client = OpenAI(
                base_url=settings.OPENROUTER_BASE_URL,
                api_key=settings.OPENROUTER_API_KEY,
            )

    def _resolve_client(self, api_key_override: str | None) -> OpenAI | None:
        """Resolve request-scoped or default OpenRouter client."""
        if api_key_override:
            return OpenAI(
                base_url=settings.OPENROUTER_BASE_URL,
                api_key=api_key_override,
            )
        return self._client

    async def answer(
        self,
        question: str,
        history: list[ChatHistoryItem] | None = None,
        api_key_override: str | None = None,
    ) -> QuestionResponse:
        """Return an answer from OpenRouter or a local fallback in test mode."""
        history = history or []
        client = self._resolve_client(api_key_override)

        if client is None:
            answer = (
                "Respuesta en modo prueba (sin API key). "
                f"Pregunta: '{question}'. "
                "Configura OPENROUTER_API_KEY para obtener respuestas del modelo."
            )
            return QuestionResponse(answer=answer, source_hint="mock://no-api-key")

        try:
            messages: list[dict[str, str]] = [
                {
                    "role": "system",
                    "content": (
                        "Eres un asistente experto en regulacion energetica colombiana. "
                        "Responde siempre en español claro y tecnico. "
                        "Si falta contexto, dilo explicitamente y propone que informacion adicional se requiere."
                    ),
                }
            ]
            for item in history[-12:]:
                messages.append({"role": item.role, "content": item.content})
            messages.append({"role": "user", "content": question})

            completion = await asyncio.to_thread(
                client.chat.completions.create,
                model=settings.OPENROUTER_CHAT_MODEL,
                messages=messages,
                extra_body={"reasoning": {"enabled": True}},
            )
            answer = completion.choices[0].message.content or "El modelo retorno una respuesta vacia."
            return QuestionResponse(answer=answer, source_hint="openrouter://chat")
        except Exception:
            answer = (
                "Respuesta de contingencia. "
                f"Pregunta: '{question}'. "
                "La llamada a OpenRouter fallo, pero el endpoint sigue operativo."
            )
            return QuestionResponse(answer=answer, source_hint="mock://chat-fallback")

    async def stream_answer(
        self,
        question: str,
        history: list[ChatHistoryItem] | None = None,
        api_key_override: str | None = None,
    ):
        """Stream answer chunks from OpenRouter when available, with fallback chunking."""
        history = history or []
        client = self._resolve_client(api_key_override)

        if client is None:
            fallback = (
                "Respuesta en modo prueba (sin API key). "
                f"Pregunta: '{question}'. "
                "Configura OPENROUTER_API_KEY para obtener respuestas del modelo."
            )
            for chunk in self._chunk_text(fallback):
                yield chunk
            return

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "Eres un asistente experto en regulacion energetica colombiana. "
                    "Responde siempre en español claro y tecnico. "
                    "Si falta contexto, dilo explicitamente y propone que informacion adicional se requiere."
                ),
            }
        ]
        for item in history[-12:]:
            messages.append({"role": item.role, "content": item.content})
        messages.append({"role": "user", "content": question})

        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:
                stream = client.chat.completions.create(
                    model=settings.OPENROUTER_CHAT_MODEL,
                    messages=messages,
                    extra_body={"reasoning": {"enabled": True}},
                    stream=True,
                )
                for event in stream:
                    delta = event.choices[0].delta.content
                    if delta:
                        loop.call_soon_threadsafe(queue.put_nowait, delta)
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, exc)

        threading.Thread(target=worker, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                fallback = (
                    "Respuesta de contingencia. "
                    f"Pregunta: '{question}'. "
                    "La llamada a OpenRouter fallo, pero el endpoint sigue operativo."
                )
                for chunk in self._chunk_text(fallback):
                    yield chunk
                break
            yield item

    def _chunk_text(self, text: str, size: int = 80) -> list[str]:
        """Split text into small chunks for fallback streaming."""
        return [text[i : i + size] for i in range(0, len(text), size)]
