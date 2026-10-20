import asyncio

import httpx
import pytest

from app.corpus import fetch as fetch_module
from app.corpus import text as text_module
from app.corpus.fetch import FetchError, PoliteFetcher
from app.corpus.storage import LocalBlobStorage


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    async def instant(_seconds):
        return None

    monkeypatch.setattr(fetch_module.asyncio, "sleep", instant)


def run(coroutine):
    return asyncio.run(coroutine)


def test_fetcher_sends_validators_and_reports_not_modified() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(304, headers={"ETag": '"a"'})

    async def scenario():
        fetcher = PoliteFetcher(transport=httpx.MockTransport(handler))
        try:
            return await fetcher.get("https://example.gov.co/x", etag='"a"', last_modified="L")
        finally:
            await fetcher.close()

    result = run(scenario())
    assert result.not_modified
    assert seen["if-none-match"] == '"a"'
    assert seen["if-modified-since"] == "L"
    assert seen["user-agent"].startswith("EnerColBot")


def test_fetcher_retries_server_errors_and_dropped_connections() -> None:
    attempts = iter(["drop", 503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        outcome = next(attempts)
        if outcome == "drop":
            raise httpx.ConnectError("connection reset")
        return httpx.Response(outcome, content=b"ok")

    async def scenario():
        fetcher = PoliteFetcher(transport=httpx.MockTransport(handler))
        try:
            return await fetcher.get("https://example.gov.co/x")
        finally:
            await fetcher.close()

    assert run(scenario()).content == b"ok"


def test_fetcher_does_not_retry_client_errors() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(404)

    async def scenario():
        fetcher = PoliteFetcher(transport=httpx.MockTransport(handler))
        try:
            await fetcher.get("https://example.gov.co/missing")
        finally:
            await fetcher.close()

    with pytest.raises(FetchError, match="HTTP 404"):
        run(scenario())
    assert len(calls) == 1


def test_storage_is_content_addressed_and_confined(tmp_path) -> None:
    storage = LocalBlobStorage(tmp_path)
    key = run(storage.put(b"contenido", ".PDF"))
    assert key.endswith(".pdf")
    assert run(storage.put(b"contenido", ".pdf")) == key
    assert run(storage.get(key)) == b"contenido"
    with pytest.raises(ValueError):
        run(storage.get("../../etc/passwd"))


def test_scanned_pdfs_fall_back_to_ocr(monkeypatch) -> None:
    class FakePage:
        def extract_text(self):
            return "Re"

    class FakeReader:
        def __init__(self, *_args, **_kwargs):
            self.pages = [FakePage(), FakePage()]

    monkeypatch.setattr(text_module, "PdfReader", FakeReader)
    monkeypatch.setattr(text_module, "ocr_pdf", lambda _content: "ARTÍCULO 1. Texto escaneado")
    assert text_module.pdf_text(b"%PDF") == ("ARTÍCULO 1. Texto escaneado", "ocr")


def test_enqueueing_already_queued_work_is_not_an_error() -> None:
    from pgqueuer.errors import DuplicateJobError

    from app.corpus.worker import make_enqueue

    class QueueWithExistingJob:
        def __init__(self):
            self.calls = []

        async def enqueue(self, entrypoint, payload, dedupe_key):
            self.calls.append(dedupe_key)
            raise DuplicateJobError(dedupe_key)

    queue = QueueWithExistingJob()
    run(make_enqueue(queue)("corpus.process", {"norm_id": "n1"}, "n1"))
    assert queue.calls == ["corpus.process:n1"]
