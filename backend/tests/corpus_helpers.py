import gzip
from pathlib import Path

from app.corpus.fetch import FetchError, FetchResult
from app.corpus.sources import creg

FIXTURES = Path(__file__).parent / "fixtures" / "creg"


def fixture(name: str) -> bytes:
    return gzip.decompress((FIXTURES / f"{name}.gz").read_bytes())


def doc_url(name: str) -> str:
    return f"{creg.BASE_URL}docs/{name}"


def listing_url(name: str) -> str:
    return f"{creg.BASE_URL}{name}"


class FakeFetcher:
    """Serves fixture pages by URL and answers conditional requests like the CREG's IIS."""

    def __init__(self, pages: dict[str, bytes]) -> None:
        self.pages = dict(pages)
        self.versions: dict[str, int] = {url: 1 for url in pages}
        self.requests: list[str] = []

    def replace(self, url: str, content: bytes) -> None:
        self.pages[url] = content
        self.versions[url] = self.versions.get(url, 0) + 1

    async def get(self, url, etag=None, last_modified=None) -> FetchResult:
        self.requests.append(url)
        if url.endswith(".pdf"):
            return FetchResult(url, 200, b"%PDF-1.4 fake", "application/pdf", None, None)
        if url not in self.pages:
            raise FetchError(f"{url} returned HTTP 404")
        current = f'"v{self.versions[url]}"'
        if etag == current:
            return FetchResult(url, 304, b"", None, current, None)
        return FetchResult(url, 200, self.pages[url], "text/html", current, None)

    async def close(self) -> None:
        pass
