import logging
from collections.abc import AsyncGenerator
from typing import Any

from pgvector.asyncpg import register_vector
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import DeclarativeBase, MappedAsDataclass, sessionmaker
from sqlalchemy.util.concurrency import await_only

from ..config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase, MappedAsDataclass):
    pass


DATABASE_URI = settings.POSTGRES_URI
DATABASE_PREFIX = settings.POSTGRES_ASYNC_PREFIX
DATABASE_URL = f"{DATABASE_PREFIX}{DATABASE_URI}"

logger.info("初始化异步数据库引擎: %s:%s/%s", settings.POSTGRES_SERVER, settings.POSTGRES_PORT, settings.POSTGRES_DB)

async_engine = create_async_engine(
    DATABASE_URL,
    echo=settings.DB_ECHO,
    future=True,
    # 关闭 SQLAlchemy 2.0 的 insertmanyvalues 批量插入优化：
    # 该特性会把多条 INSERT 拼成 "INSERT ... SELECT p0::VECTOR(1536) ..."，
    # 带显式 CAST 的参数绑定会绕过 asyncpg 的 pgvector 二进制 codec，
    # 导致 embedding 的 list 无法编码（expected str, got list）。
    # 关闭后所有插入走单条绑定路径，与 PDF 入库行为一致。
    use_insertmanyvalues=False,
)


@event.listens_for(async_engine.sync_engine, "connect")
def _register_pgvector_codec(dbapi_connection, connection_record) -> None:
    """在每次新建数据库连接时注册 pgvector 的 asyncpg 二进制编解码器。

    models.Vector 类型为了让 list 原样交给 asyncpg 编码而禁用了 SQLAlchemy
    层的字符串转换，因此必须依赖 asyncpg 侧的 register_vector 二进制 codec，
    否则插入 embedding 时会报 "expected str, got list"。
    """
    try:
        # connect 事件收到的是 SQLAlchemy 的 DBAPI 适配器连接，需取出其内部的
        # 原生 asyncpg Connection 才能调用 set_type_codec。
        asyncpg_connection = dbapi_connection._connection
        # register_vector 是 async 函数；connect 事件在 SQLAlchemy 的 greenlet
        # 上下文中触发，可用 await_only 同步驱动它完成 asyncpg 层的 codec 注册。
        await_only(register_vector(asyncpg_connection))
    except Exception:
        # 若目标库未启用 pgvector 扩展则跳过注册，不影响非向量功能。
        logger.warning("pgvector codec 注册失败（数据库可能未启用 vector 扩展）", exc_info=True)


local_session = sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)  # type: ignore[call-overload]


async def async_get_db() -> AsyncGenerator[Any, Any]:
    """创建并管理异步数据库会话的依赖，请求结束后自动关闭会话。"""
    logger.debug("创建数据库会话")
    async with local_session() as db:
        yield db
    logger.debug("关闭数据库会话")
