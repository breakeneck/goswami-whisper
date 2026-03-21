"""Initial migration

Revision ID: b5b54a07fa5d
Revises: 
Create Date: 2026-03-21 05:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = 'b5b54a07fa5d'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    inspector = inspect(conn)

    # Create uploads table if not exists
    if not inspector.has_table('uploads'):
        op.create_table('uploads',
            sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
            sa.Column('filename', sa.String(length=500), nullable=False),
            sa.Column('original_filename', sa.String(length=500), nullable=True),
            sa.Column('file_path', sa.String(length=1000), nullable=True),
            sa.Column('source_url', sa.Text(), nullable=True),
            sa.Column('file_size', sa.BigInteger(), nullable=True),
            sa.Column('duration_seconds', sa.Float(), nullable=True),
            sa.Column('is_indexed', sa.Boolean(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )

    # Create transcribes table if not exists
    if not inspector.has_table('transcribes'):
        op.create_table('transcribes',
            sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
            sa.Column('upload_id', sa.Integer(), nullable=False),
            sa.Column('text', mysql.LONGTEXT(), nullable=True),
            sa.Column('provider', sa.String(length=50), nullable=False),
            sa.Column('model', sa.String(length=50), nullable=False),
            sa.Column('status', sa.String(length=50), nullable=True),
            sa.Column('progress', sa.Float(), nullable=True),
            sa.Column('error_message', sa.Text(), nullable=True),
            sa.Column('duration_seconds', sa.Float(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['upload_id'], ['uploads.id'], ),
            sa.PrimaryKeyConstraint('id')
        )

    # Create contents table if not exists
    if not inspector.has_table('contents'):
        op.create_table('contents',
            sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
            sa.Column('transcribe_id', sa.Integer(), nullable=False),
            sa.Column('text', mysql.LONGTEXT(), nullable=True),
            sa.Column('provider', sa.String(length=50), nullable=False),
            sa.Column('model', sa.String(length=100), nullable=False),
            sa.Column('status', sa.String(length=50), nullable=True),
            sa.Column('progress', sa.Float(), nullable=True),
            sa.Column('error_message', sa.Text(), nullable=True),
            sa.Column('is_indexed', sa.Boolean(), nullable=True),
            sa.Column('duration_seconds', sa.Float(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['transcribe_id'], ['transcribes.id'], ),
            sa.PrimaryKeyConstraint('id')
        )


def downgrade():
    op.drop_table('contents')
    op.drop_table('transcribes')
    op.drop_table('uploads')
