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
    "minenergia": SourceInfo(
        id="minenergia",
        name="MinEnergía",
        full_name="Ministerio de Minas y Energía",
        url="https://normativame.minenergia.gov.co/",
        doc_types={
            "resolucion": "Resolución",
            "decreto": "Decreto",
            "ley": "Ley",
            "circular": "Circular",
            "proyecto_resolucion": "Proyecto de Resolución",
            "proyecto_decreto": "Proyecto de Decreto",
        },
    ),
}

# Norms referenced by amendments but issued outside the crawled sources.
EXTERNAL_ISSUERS = {
    "co": "",
    "upme": "UPME",
}
# Signed nationally even when a ministry publishes them, so they carry no issuer.
NATIONAL_DOC_TYPES = {"ley", "decreto", "proyecto_decreto"}
GENERIC_DOC_TYPES = {
    "ley": "Ley",
    "decreto": "Decreto",
    "resolucion": "Resolución",
    "proyecto_resolucion": "Proyecto de Resolución",
    "proyecto_decreto": "Proyecto de Decreto",
    "circular": "Circular",
}


def norm_label(source: str, doc_type: str, number: str, year: int, topic: str = "") -> str:
    """'Resolución CREG 101 99 de 2026', 'Decreto 1073 de 2015'.

    Unnumbered drafts are told apart by their topic instead.
    """
    info = SOURCES.get(source)
    kind = (info.doc_types.get(doc_type) if info else None) or GENERIC_DOC_TYPES.get(
        doc_type, doc_type.replace("_", " ").capitalize()
    )
    if doc_type in NATIONAL_DOC_TYPES:
        issuer = ""
    else:
        issuer = info.name if info else EXTERNAL_ISSUERS.get(source, source.upper())
    if not number and topic:
        return f"{' '.join(part for part in (kind, issuer) if part)}: {topic}"
    parts = [kind, issuer, number.replace("-", " "), "de", str(year)]
    return " ".join(part for part in parts if part)


def key_label(key: str) -> str:
    """Label for a norm key that may not be in the corpus, e.g. 'co:decreto:1073:2015'."""
    try:
        source, doc_type, number, year = key.split(":")
        return norm_label(source, doc_type, number, int(year))
    except ValueError:
        return key
