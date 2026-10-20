from app.corpus.sources import creg, minenergia
from app.corpus.sources.base import Source

ADAPTERS: dict[str, Source] = {
    adapter.id: adapter for adapter in (creg.ADAPTER, minenergia.ADAPTER)
}


def adapter_for(source: str) -> Source:
    try:
        return ADAPTERS[source]
    except KeyError:
        raise ValueError(f"Unknown corpus source: {source}") from None
