"""Regulatory corpus: norms, versions, files, searchable articles, relations, events, crawls."""

import sqlalchemy as sa
from pgvector.sqlalchemy import HALFVEC
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0004_regulatory_corpus"
down_revision = "0003_pplx_embeddings_1024"
branch_labels = None
depends_on = None

EMBEDDING_DIMENSIONS = 1024


def upgrade() -> None:
    op.create_table(
        "norms",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("key", sa.String(length=200), nullable=False, unique=True),
        sa.Column("source", sa.String(length=40), nullable=False),
        sa.Column("doc_type", sa.String(length=40), nullable=False),
        sa.Column("number", sa.String(length=40), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("issued_on", sa.Date(), nullable=True),
        sa.Column("published_on", sa.Date(), nullable=True),
        sa.Column("diario_oficial", sa.String(length=80), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("etag", sa.String(length=200), nullable=True),
        sa.Column("last_modified", sa.String(length=80), nullable=True),
        sa.Column("content_sha256", sa.String(length=64), nullable=True),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_changed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("extra", postgresql.JSONB(), nullable=False),
    )
    op.create_index("ix_norms_source", "norms", ["source"])
    op.create_index("ix_norms_doc_type", "norms", ["doc_type"])
    op.create_index("ix_norms_year", "norms", ["year"])
    op.create_index("ix_norms_issued_on", "norms", ["issued_on"])
    op.create_index("ix_norms_status", "norms", ["status"])

    op.create_table(
        "norm_versions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("norm_id", sa.String(length=36), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("text_source", sa.String(length=20), nullable=False),
        sa.Column("raw_storage_key", sa.Text(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["norm_id"], ["norms.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("norm_id", "content_sha256"),
    )
    op.create_index("ix_norm_versions_norm_id", "norm_versions", ["norm_id"])

    op.create_table(
        "norm_files",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("norm_id", sa.String(length=36), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("label", sa.String(length=200), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("content_type", sa.String(length=120), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("storage_key", sa.Text(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["norm_id"], ["norms.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("norm_id", "url"),
    )
    op.create_index("ix_norm_files_norm_id", "norm_files", ["norm_id"])

    op.create_table(
        "norm_articles",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("norm_id", sa.String(length=36), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(length=200), nullable=False),
        sa.Column("norm_label", sa.String(length=300), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", HALFVEC(EMBEDDING_DIMENSIONS), nullable=True),
        sa.Column(
            "search",
            postgresql.TSVECTOR(),
            sa.Computed(
                "setweight(to_tsvector('spanish', coalesce(norm_label, '') || ' ' || "
                "coalesce(label, '')), 'A') || to_tsvector('spanish', coalesce(content, ''))",
                persisted=True,
            ),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["norm_id"], ["norms.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_norm_articles_norm_id", "norm_articles", ["norm_id"])
    op.create_index("ix_norm_articles_search", "norm_articles", ["search"], postgresql_using="gin")
    op.execute(
        "CREATE INDEX ix_norm_articles_embedding ON norm_articles "
        "USING hnsw (embedding halfvec_cosine_ops)"
    )

    op.create_table(
        "norm_relations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("from_key", sa.String(length=200), nullable=False),
        sa.Column("to_key", sa.String(length=200), nullable=False),
        sa.Column("relation", sa.String(length=30), nullable=False),
        sa.Column("from_article", sa.String(length=80), nullable=False),
        sa.Column("to_article", sa.String(length=80), nullable=False),
        sa.Column("evidence", sa.Text(), nullable=False),
        sa.Column("reported_by_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["reported_by_id"], ["norms.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "from_key", "to_key", "relation", "from_article", "to_article", name="uq_norm_relation"
        ),
    )
    op.create_index("ix_norm_relations_to_key", "norm_relations", ["to_key"])
    op.create_index("ix_norm_relations_from_key", "norm_relations", ["from_key"])
    op.create_index("ix_norm_relations_reported_by_id", "norm_relations", ["reported_by_id"])

    op.create_table(
        "norm_events",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("norm_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=30), nullable=False),
        sa.Column("details", postgresql.JSONB(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["norm_id"], ["norms.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_norm_events_norm_id", "norm_events", ["norm_id"])
    op.create_index("ix_norm_events_kind", "norm_events", ["kind"])
    op.create_index("ix_norm_events_occurred_at", "norm_events", ["occurred_at"])

    op.create_table(
        "crawl_runs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("source", sa.String(length=40), nullable=False),
        sa.Column("mode", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("pages_fetched", sa.Integer(), nullable=False),
        sa.Column("pages_unchanged", sa.Integer(), nullable=False),
        sa.Column("items_seen", sa.Integer(), nullable=False),
        sa.Column("items_queued", sa.Integer(), nullable=False),
        sa.Column("errors", sa.Integer(), nullable=False),
        sa.Column("error_sample", sa.Text(), nullable=True),
    )
    op.create_index("ix_crawl_runs_source", "crawl_runs", ["source"])
    op.create_index("ix_crawl_runs_started_at", "crawl_runs", ["started_at"])

    op.create_table(
        "fetch_cache",
        sa.Column("url", sa.Text(), primary_key=True),
        sa.Column("etag", sa.String(length=200), nullable=True),
        sa.Column("last_modified", sa.String(length=80), nullable=True),
        sa.Column("content_sha256", sa.String(length=64), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    for table in (
        "fetch_cache",
        "crawl_runs",
        "norm_events",
        "norm_relations",
        "norm_articles",
        "norm_files",
        "norm_versions",
        "norms",
    ):
        op.drop_table(table)
