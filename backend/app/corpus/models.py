from datetime import date, datetime
from typing import Any

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import (
    Computed,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.models import Base, new_id, utc_now


class Norm(Base):
    """One regulatory act from an official source, identified across re-crawls by its key."""

    __tablename__ = "norms"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    # e.g. "creg:resolucion:101-127:2026"; stable even when the source URL changes.
    key: Mapped[str] = mapped_column(String(200), unique=True)
    source: Mapped[str] = mapped_column(String(40), index=True)
    doc_type: Mapped[str] = mapped_column(String(40), index=True)
    number: Mapped[str] = mapped_column(String(40))
    year: Mapped[int] = mapped_column(Integer, index=True)
    title: Mapped[str] = mapped_column(Text, default="")
    issued_on: Mapped[date | None] = mapped_column(Date, index=True)
    published_on: Mapped[date | None] = mapped_column(Date)
    diario_oficial: Mapped[str | None] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(20), default="desconocido", index=True)
    url: Mapped[str] = mapped_column(Text)
    etag: Mapped[str | None] = mapped_column(String(200))
    last_modified: Mapped[str | None] = mapped_column(String(80))
    content_sha256: Mapped[str | None] = mapped_column(String(64))
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    extra: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    articles: Mapped[list["NormArticle"]] = relationship(
        cascade="all, delete-orphan", passive_deletes=True, order_by="NormArticle.position"
    )
    files: Mapped[list["NormFile"]] = relationship(
        cascade="all, delete-orphan", passive_deletes=True
    )


class NormVersion(Base):
    """Every distinct text seen for a norm, so amendments can be compared over time."""

    __tablename__ = "norm_versions"
    __table_args__ = (UniqueConstraint("norm_id", "content_sha256"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    norm_id: Mapped[str] = mapped_column(ForeignKey("norms.id", ondelete="CASCADE"), index=True)
    content_sha256: Mapped[str] = mapped_column(String(64))
    text: Mapped[str] = mapped_column(Text)
    text_source: Mapped[str] = mapped_column(String(20))
    raw_storage_key: Mapped[str | None] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class NormFile(Base):
    __tablename__ = "norm_files"
    __table_args__ = (UniqueConstraint("norm_id", "url"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    norm_id: Mapped[str] = mapped_column(ForeignKey("norms.id", ondelete="CASCADE"), index=True)
    url: Mapped[str] = mapped_column(Text)
    label: Mapped[str] = mapped_column(String(200), default="")
    kind: Mapped[str] = mapped_column(String(20))
    content_type: Mapped[str | None] = mapped_column(String(120))
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    sha256: Mapped[str | None] = mapped_column(String(64))
    storage_key: Mapped[str | None] = mapped_column(Text)
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class NormArticle(Base):
    """A searchable passage: one article, or one part of a long article."""

    __tablename__ = "norm_articles"
    __table_args__ = (
        Index("ix_norm_articles_search", "search", postgresql_using="gin"),
        Index(
            "ix_norm_articles_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "halfvec_cosine_ops"},
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    norm_id: Mapped[str] = mapped_column(ForeignKey("norms.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer)
    label: Mapped[str] = mapped_column(String(200))
    # Denormalized "Resolución CREG 174 de 2021" so lexical search matches citations by number.
    norm_label: Mapped[str] = mapped_column(String(300))
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(HALFVEC(settings.EMBEDDING_DIMENSIONS))
    search: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('spanish', coalesce(norm_label, '') || ' ' || "
            "coalesce(label, '')), 'A') || to_tsvector('spanish', coalesce(content, ''))",
            persisted=True,
        ),
    )


class NormRelation(Base):
    """A directed edge such as "Res. 101-99/2026 art. 50 modifica Res. 174/2021 art. 1".

    The target is kept as a key because it may not be crawled (yet) or may belong to another
    source, e.g. a national decree.
    """

    __tablename__ = "norm_relations"
    __table_args__ = (
        UniqueConstraint(
            "from_key", "to_key", "relation", "from_article", "to_article", name="uq_norm_relation"
        ),
        Index("ix_norm_relations_to_key", "to_key"),
        Index("ix_norm_relations_from_key", "from_key"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    from_key: Mapped[str] = mapped_column(String(200))
    to_key: Mapped[str] = mapped_column(String(200))
    relation: Mapped[str] = mapped_column(String(30))
    from_article: Mapped[str] = mapped_column(String(80), default="")
    to_article: Mapped[str] = mapped_column(String(80), default="")
    evidence: Mapped[str] = mapped_column(Text, default="")
    # The norm whose page stated this relation; re-crawling it replaces its edges.
    reported_by_id: Mapped[str] = mapped_column(
        ForeignKey("norms.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class NormEvent(Base):
    """Feed of changes in the corpus: new norms, new versions, status changes."""

    __tablename__ = "norm_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    norm_id: Mapped[str] = mapped_column(ForeignKey("norms.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(30), index=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True
    )


class CrawlRun(Base):
    __tablename__ = "crawl_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    source: Mapped[str] = mapped_column(String(40), index=True)
    mode: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="running")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    pages_fetched: Mapped[int] = mapped_column(Integer, default=0)
    pages_unchanged: Mapped[int] = mapped_column(Integer, default=0)
    items_seen: Mapped[int] = mapped_column(Integer, default=0)
    items_queued: Mapped[int] = mapped_column(Integer, default=0)
    errors: Mapped[int] = mapped_column(Integer, default=0)
    error_sample: Mapped[str | None] = mapped_column(Text)


class FetchCache(Base):
    """Validators for listing pages, so periodic crawls send conditional requests."""

    __tablename__ = "fetch_cache"

    url: Mapped[str] = mapped_column(Text, primary_key=True)
    etag: Mapped[str | None] = mapped_column(String(200))
    last_modified: Mapped[str | None] = mapped_column(String(80))
    content_sha256: Mapped[str | None] = mapped_column(String(64))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
