"""Create validation_rules.

Rules move out of YAML playbook files and into the database, so policy owners
can edit them through the API without a release, and so they can be enabled,
disabled and scoped per document type.

Revision ID: 0001_validation_rules
Revises:
Create Date: 2026-09-29
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_validation_rules"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# The same type shims the model uses - a migration that declares different
# column types than the ORM binds produces errors only on one backend.
JSON_TYPE = postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite")
GUID_TYPE = postgresql.UUID(as_uuid=True).with_variant(sa.String(length=36), "sqlite")


def upgrade() -> None:
    op.create_table(
        "validation_rules",
        sa.Column("id", GUID_TYPE, nullable=False),
        sa.Column("code", sa.String(length=64), nullable=False),
        sa.Column("doctype", sa.String(length=64), nullable=True),
        sa.Column("group_name", sa.String(length=128), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("rule", sa.Text(), nullable=False),
        sa.Column("fallback", sa.Text(), nullable=True),
        sa.Column("preferred_language", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("severity", sa.String(length=16), nullable=False, server_default="medium"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("sort_order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("keywords", JSON_TYPE, nullable=True),
        sa.Column("extra", JSON_TYPE, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "severity IN ('low', 'medium', 'high')", name="chk_validation_rules_severity"
        ),
    )
    op.create_index("ix_validation_rules_code", "validation_rules", ["code"])
    op.create_index("ix_validation_rules_doctype", "validation_rules", ["doctype"])
    op.create_index("ix_validation_rules_group_name", "validation_rules", ["group_name"])
    op.create_index("ix_validation_rules_enabled", "validation_rules", ["enabled"])
    # One code per document type; the same code may exist for another type.
    op.create_index(
        "uq_validation_rules_doctype_code",
        "validation_rules",
        ["doctype", "code"],
        unique=True,
    )
    op.create_index(
        "idx_validation_rules_doctype_enabled", "validation_rules", ["doctype", "enabled"]
    )


def downgrade() -> None:
    op.drop_index("idx_validation_rules_doctype_enabled", table_name="validation_rules")
    op.drop_index("uq_validation_rules_doctype_code", table_name="validation_rules")
    op.drop_index("ix_validation_rules_enabled", table_name="validation_rules")
    op.drop_index("ix_validation_rules_group_name", table_name="validation_rules")
    op.drop_index("ix_validation_rules_doctype", table_name="validation_rules")
    op.drop_index("ix_validation_rules_code", table_name="validation_rules")
    op.drop_table("validation_rules")
