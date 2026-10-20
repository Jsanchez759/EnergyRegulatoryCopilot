import asyncio
import logging
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class FetchError(RuntimeError):
    """Raised when a source keeps failing after retries."""


@dataclass(frozen=True)
class FetchResult:
    url: str
    status: int
    content: bytes
    content_type: str | None
    etag: str | None
    last_modified: str | None

    @property
    def not_modified(self) -> bool:
        return self.status == 304


class PoliteFetcher:
    """HTTP client that spaces requests per host and retries transient failures.

    Government sites drop connections under bursts instead of answering 429, so pacing is
    enforced here rather than trusted to callers.
    """

    def __init__(
        self,
        min_interval_by_host: dict[str, float] | None = None,
        default_min_interval: float = 2.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._intervals = min_interval_by_host or {}
        self._default_interval = default_min_interval
        self._last_request: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._client = httpx.AsyncClient(
            headers={"User-Agent": settings.CRAWLER_USER_AGENT, "Accept-Encoding": "gzip"},
            timeout=settings.CRAWLER_TIMEOUT_SECONDS,
            follow_redirects=True,
            transport=transport,
        )

    async def get(
        self, url: str, etag: str | None = None, last_modified: str | None = None
    ) -> FetchResult:
        headers = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        host = urlsplit(url).hostname or ""
        lock = self._locks.setdefault(host, asyncio.Lock())
        last_error: Exception | None = None
        for attempt in range(settings.CRAWLER_MAX_RETRIES + 1):
            async with lock:
                await self._wait_turn(host)
                try:
                    response = await self._client.get(url, headers=headers)
                except httpx.TransportError as exc:
                    last_error = exc
                    response = None
                finally:
                    self._last_request[host] = time.monotonic()
            if response is not None and response.status_code < 500:
                if response.status_code >= 400:
                    raise FetchError(f"{url} returned HTTP {response.status_code}")
                return FetchResult(
                    url=str(response.url),
                    status=response.status_code,
                    content=response.content,
                    content_type=response.headers.get("content-type"),
                    etag=response.headers.get("etag"),
                    last_modified=response.headers.get("last-modified"),
                )
            if response is not None:
                last_error = FetchError(f"{url} returned HTTP {response.status_code}")
            backoff = min(60.0, 5.0 * 2**attempt)
            logger.warning(
                "fetch failed, backing off",
                extra={"url": url, "attempt": attempt + 1, "backoff_s": backoff},
            )
            await asyncio.sleep(backoff)
        raise FetchError(f"{url} failed after retries: {last_error}") from last_error

    async def _wait_turn(self, host: str) -> None:
        interval = self._intervals.get(host, self._default_interval)
        elapsed = time.monotonic() - self._last_request.get(host, 0.0)
        if elapsed < interval:
            await asyncio.sleep(interval - elapsed)

    async def close(self) -> None:
        await self._client.aclose()
