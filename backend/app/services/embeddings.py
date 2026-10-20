import asyncio
import logging
import time

from openai import AsyncOpenAI, OpenAIError

from app.core.config import settings
from app.services.errors import ProviderUnavailableError

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Batched, concurrent embedding requests with dimension checks against the schema."""

    def __init__(self) -> None:
        self._client = (
            AsyncOpenAI(
                base_url=settings.OPENROUTER_BASE_URL,
                api_key=settings.OPENROUTER_API_KEY,
                timeout=settings.EMBEDDING_TIMEOUT_SECONDS,
                max_retries=settings.EMBEDDING_MAX_RETRIES,
            )
            if settings.OPENROUTER_API_KEY
            else None
        )

    @property
    def available(self) -> bool:
        return self._client is not None

    def require_available(self) -> None:
        if self._client is None:
            raise ProviderUnavailableError("OPENROUTER_API_KEY is not configured on the server.")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.require_available()
        client = self._client
        batch_size = settings.EMBEDDING_BATCH_SIZE
        batches = [texts[start : start + batch_size] for start in range(0, len(texts), batch_size)]
        # Provider stalls are per request, so concurrent batches keep one stall from blocking all.
        semaphore = asyncio.Semaphore(settings.EMBEDDING_CONCURRENCY)

        async def embed_batch(batch: list[str]) -> list[list[float]]:
            async with semaphore:
                response = await client.embeddings.create(
                    model=settings.OPENROUTER_EMBEDDING_MODEL, input=batch
                )
            return [item.embedding for item in response.data]

        started = time.perf_counter()
        try:
            results = await asyncio.gather(*(embed_batch(batch) for batch in batches))
        except OpenAIError as exc:
            logger.warning("embedding provider request failed: %s", type(exc).__name__)
            raise ProviderUnavailableError(
                "The embedding provider is not available right now."
            ) from exc
        embeddings = [vector for batch in results for vector in batch]
        logger.info(
            "embedded texts",
            extra={
                "texts": len(texts),
                "batches": len(batches),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        if any(len(vector) != settings.EMBEDDING_DIMENSIONS for vector in embeddings):
            logger.error(
                "embedding model %s returned vectors that do not match EMBEDDING_DIMENSIONS=%s",
                settings.OPENROUTER_EMBEDDING_MODEL,
                settings.EMBEDDING_DIMENSIONS,
            )
            raise ProviderUnavailableError("The embedding model is misconfigured on the server.")
        return embeddings

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
