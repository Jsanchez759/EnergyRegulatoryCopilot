"""Switch chunk embeddings to 1024 dimensions for perplexity/pplx-embed-v1-0.6b.

Vectors from different embedding models are not comparable and uploads are not stored, so
existing documents are removed and must be uploaded again.
"""

from pgvector.sqlalchemy import HALFVEC

from alembic import op

revision = "0003_pplx_embeddings_1024"
down_revision = "0002_documents_pgvector"
branch_labels = None
depends_on = None


def _replace_embedding_column(dimensions: int) -> None:
    op.execute("DELETE FROM documents")
    op.drop_index("ix_document_chunks_embedding", table_name="document_chunks")
    op.alter_column(
        "document_chunks",
        "embedding",
        type_=HALFVEC(dimensions),
        postgresql_using=f"embedding::halfvec({dimensions})",
    )
    op.execute(
        "CREATE INDEX ix_document_chunks_embedding ON document_chunks "
        "USING hnsw (embedding halfvec_cosine_ops)"
    )


def upgrade() -> None:
    _replace_embedding_column(1024)


def downgrade() -> None:
    _replace_embedding_column(2048)
