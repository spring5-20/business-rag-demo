import uuid as uuid_pkg
from datetime import UTC, datetime
from typing import Any

from pgvector.sqlalchemy import Vector as PgVectorType
from sqlalchemy import (
    BigInteger,
    Boolean,
    Computed,
    DateTime,
    Dialect,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class Vector(PgVectorType):
    """适配 asyncpg 驱动的 pgvector 向量列类型。

    pgvector 自带的 SQLAlchemy 类型会把 Python list 转成字符串后再绑定，
    而 asyncpg 的 pgvector 编解码器需要 list，因此这里禁用字符串转换，
    让 list 原样交给 asyncpg 编码。
    """

    def bind_processor(self, dialect: Dialect) -> None:
        return None


class UUIDMixin:
    uuid: Mapped[uuid_pkg.UUID] = mapped_column(
        Uuid, primary_key=True, default_factory=uuid_pkg.uuid4, server_default=text("gen_random_uuid()")
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default_factory=lambda: datetime.now(UTC), server_default=text("current_timestamp(0)")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=True,
        onupdate=lambda: datetime.now(UTC),
        server_default=text("current_timestamp(0)"),
    )


class SoftDeleteMixin:
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))


class ConnectionTest(Base):
    """连接验证表，用于在 PostgreSQL 连接成功时自动创建并验证连接可用。"""

    __tablename__ = "connection_test"

    id: Mapped[uuid_pkg.UUID] = mapped_column(
        Uuid, primary_key=True, default_factory=uuid_pkg.uuid4, server_default=text("gen_random_uuid()")
    )
    note: Mapped[str] = mapped_column(String(255), default="connection-ok")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default_factory=lambda: datetime.now(UTC), server_default=text("current_timestamp(0)")
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default=text("true"))


class RagDocument(Base):
    """RAG 文档主表：记录每份入库的原始文档（如 PDF）的元信息。"""

    __tablename__ = "rag_documents"

    # 必填业务字段（无 Python 默认值，构造时必须传入，需排在带默认值字段之前）
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # 可选字段：给出 Python 默认值，避免 MappedAsDataclass 把 nullable 列当必填
    source_path: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    author: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default_factory=dict, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default_factory=lambda: datetime.now(UTC), server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default_factory=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        server_default=text("now()"),
    )
    # 自增主键：init=False 使其不进入构造器，由数据库生成
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, init=False)


class RagConversation(Base):
    """RAG 会话消息表：存储用户与助手的问答消息记录。"""

    __tablename__ = "rag_conversations"

    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default_factory=lambda: datetime.now(UTC), server_default=text("now()")
    )
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, init=False)


class RagChunk(Base):
    """RAG 文档分块表：保存切分后的文本块、向量与全文检索索引。"""

    __tablename__ = "rag_chunks"

    document_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("rag_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)
    # 生成列：由 section_title / content 计算出的全文检索向量（只读，不参与写入）
    content_tsvector: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed(
            "(setweight(to_tsvector('simple'::regconfig, COALESCE(section_title, ''::text)), 'A'::\"char\") "
            "|| setweight(to_tsvector('simple'::regconfig, COALESCE(content, ''::text)), 'B'::\"char\"))",
            persisted=True,
        ),
        nullable=True,
        default=None,
    )
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    section_title: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    content_type: Mapped[str | None] = mapped_column(String(50), nullable=True, default=None)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default_factory=dict, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default_factory=lambda: datetime.now(UTC), server_default=text("now()")
    )
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, init=False)

