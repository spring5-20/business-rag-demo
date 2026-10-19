"""LangGraph 检查点生命周期管理。

默认使用 PostgreSQL 的 AsyncPostgresSaver，实现跨请求、跨进程的会话持久化；
未初始化时回退到进程内 MemorySaver，保证测试与 --skip-infra 场景仍可运行。
"""

import logging
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from .config import settings

logger = logging.getLogger(__name__)

_checkpointer: Any | None = None
_pool: Any | None = None


def get_langgraph_checkpointer() -> Any:
    """获取当前 LangGraph 检查点，未初始化时回退到 MemorySaver。"""
    global _checkpointer
    if _checkpointer is None:
        logger.warning("LangGraph 检查点未初始化，回退到进程内 MemorySaver")
        _checkpointer = MemorySaver()
    return _checkpointer


def _checkpoint_dsn() -> str:
    """构造 psycopg3 可用的 PostgreSQL 连接串。"""
    return settings.POSTGRES_URL or (
        f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )


async def init_langgraph_checkpointer() -> None:
    """初始化 PostgreSQL 检查点并创建官方 checkpoint 表。"""
    global _checkpointer, _pool

    if _checkpointer is not None and _pool is not None:
        return
    if settings.SKIP_INFRA_ON_START:
        logger.info("已跳过 LangGraph PostgreSQL 检查点（SKIP_INFRA_ON_START=true）")
        return

    dsn = _checkpoint_dsn()
    logger.info(
        "正在初始化 LangGraph PostgreSQL 检查点: %s:%s/%s",
        settings.POSTGRES_SERVER,
        settings.POSTGRES_PORT,
        settings.POSTGRES_DB,
    )
    pool: Any = AsyncConnectionPool(
        dsn,
        open=False,
        min_size=1,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open()
    try:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
    except Exception:
        await pool.close()
        raise

    _pool = pool
    _checkpointer = checkpointer
    logger.info("LangGraph PostgreSQL 检查点初始化完成")


async def close_langgraph_checkpointer() -> None:
    """关闭 LangGraph PostgreSQL 检查点连接池。"""
    global _checkpointer, _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
    _checkpointer = None