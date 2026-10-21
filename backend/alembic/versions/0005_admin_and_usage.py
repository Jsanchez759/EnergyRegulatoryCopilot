"""Account roles and waitlist status, assistant usage, and the admin audit log."""

import sqlalchemy as sa

from alembic import op

revision = "0005_admin_and_usage"
down_revision = "0004_regulatory_corpus"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users", sa.Column("role", sa.String(length=16), nullable=False, server_default="user")
    )
    op.add_column(
        "users", sa.Column("status", sa.String(length=16), nullable=False, server_default="active")
    )
    op.add_column("users", sa.Column("company", sa.String(length=150), nullable=True))
    op.add_column("users", sa.Column("profile", sa.String(length=40), nullable=True))
    op.add_column("users", sa.Column("use_case", sa.Text(), nullable=True))
    op.add_column("users", sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_users_status", "users", ["status"])

    op.create_table(
        "usage_events",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(length=36),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("feature", sa.String(length=20), nullable=False),
        sa.Column("model", sa.String(length=120), nullable=False),
        sa.Column("llm_calls", sa.Integer(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("failed", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_usage_events_user_id", "usage_events", ["user_id"])
    op.create_index("ix_usage_events_created_at", "usage_events", ["created_at"])

    op.create_table(
        "admin_audit_log",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "actor_id",
            sa.String(length=36),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("actor_email", sa.String(length=254), nullable=False),
        sa.Column("action", sa.String(length=40), nullable=False),
        sa.Column("target_user_id", sa.String(length=36), nullable=True),
        sa.Column("target_email", sa.String(length=254), nullable=True),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_admin_audit_log_target_user_id", "admin_audit_log", ["target_user_id"])
    op.create_index("ix_admin_audit_log_created_at", "admin_audit_log", ["created_at"])


def downgrade() -> None:
    op.drop_table("admin_audit_log")
    op.drop_table("usage_events")
    op.drop_index("ix_users_status", table_name="users")
    for column in ("last_login_at", "approved_at", "use_case", "profile", "company"):
        op.drop_column("users", column)
    op.drop_column("users", "status")
    op.drop_column("users", "role")
