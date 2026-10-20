"""Source-independent labels for norms, so new sources only add catalog entries."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceInfo:
    id: str
    name: str
    full_name: str
    url: str
    doc_types: dict[str, str] = field(default_factory=dict)


SOURCES: dict[str, SourceInfo] = {
    "creg": SourceInfo(
        id="creg",
        name="CREG",
        full_name="Comisión de Regulación de Energía y Gas",
        url="https://gestornormativo.creg.gov.co/gestor/",
        doc_types={
            "resolucion": "Resolución",
            "proyecto_resolucion": "Proyecto de Resolución",
            "circular": "Circular",
        },
    ),
}

# Norms referenced by amendments but issued outside the crawled sources.
EXTERNAL_ISSUERS = {
    "co": "",
    "minenergia": "MinEnergía",
    "upme": "UPME",
}
GENERIC_DOC_TYPES = {
    "ley": "Ley",
    "decreto": "Decreto",
    "resolucion": "Resolución",
    "proyecto_resolucion": "Proyecto de Resolución",
    "circular": "Circular",
}


def norm_label(source: str, doc_type: str, number: str, year: int) -> str:
    """'Resolución CREG 101 99 de 2026', 'Decreto 1073 de 2015'."""
    info = SOURCES.get(source)
    kind = (info.doc_types.get(doc_type) if info else None) or GENERIC_DOC_TYPES.get(
        doc_type, doc_type.replace("_", " ").capitalize()
    )
    issuer = info.name if info else EXTERNAL_ISSUERS.get(source, source.upper())
    parts = [kind, issuer, number.replace("-", " "), "de", str(year)]
    return " ".join(part for part in parts if part)


def key_label(key: str) -> str:
    """Label for a norm key that may not be in the corpus, e.g. 'co:decreto:1073:2015'."""
    try:
        source, doc_type, number, year = key.split(":")
        return norm_label(source, doc_type, number, int(year))
    except ValueError:
        return key
