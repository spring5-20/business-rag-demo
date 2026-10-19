import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import _AsyncGeneratorContextManager, asynccontextmanager
from typing import Any

import anyio
import fastapi
import redis.asyncio as redis
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter, FastAPI
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from ..middleware.client_cache_middleware import ClientCacheMiddleware
from .config import (
    AppSettings,
    ClientSideCacheSettings,
    DatabaseSettings,
    EnvironmentOption,
    EnvironmentSettings,
    RedisCacheSettings,
    RedisQueueSettings,
    RedisRateLimiterSettings,
    Settings,
    SkipInfraSettings,
    settings,
)
from .db import models  # noqa: F401  确保连接验证表等模型注册到 Base.metadata
from .db.database import Base
from .db.database import async_engine as engine
from .utils import cache, queue, rate_limit

logger = logging.getLogger(__name__)


# -------------- database --------------
async def create_tables() -> None:
    logger.info("正在创建/校验数据库表结构（Base.metadata.create_all）")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("数据库表结构创建/校验完成，连接验证表 connection_test 已就绪")


# -------------- cache --------------
async def create_redis_cache_pool() -> None:
    if settings.SKIP_INFRA_ON_START:
        logger.info("已跳过 Redis 缓存连接（SKIP_INFRA_ON_START=true）")
        return
    logger.info("正在连接 Redis 缓存: %s", settings.REDIS_CACHE_URL)
    cache.pool = redis.ConnectionPool.from_url(settings.REDIS_CACHE_URL)
    cache.client = redis.Redis.from_pool(cache.pool)  # type: ignore
    await cache.client.ping()
    logger.info("Redis 缓存连接成功: %s", settings.REDIS_CACHE_URL)


async def close_redis_cache_pool() -> None:
    if cache.client is not None:
        logger.info("正在关闭 Redis 缓存连接")
        await cache.client.aclose()


# -------------- queue --------------
async def create_redis_queue_pool() -> None:
    if settings.SKIP_INFRA_ON_START:
        logger.info("已跳过 Redis 队列连接（SKIP_INFRA_ON_START=true）")
        return
    logger.info("正在连接 Redis 队列: %s:%s", settings.REDIS_QUEUE_HOST, settings.REDIS_QUEUE_PORT)
    queue.pool = await create_pool(RedisSettings(host=settings.REDIS_QUEUE_HOST, port=settings.REDIS_QUEUE_PORT))
    logger.info("Redis 队列连接成功: %s:%s", settings.REDIS_QUEUE_HOST, settings.REDIS_QUEUE_PORT)


async def close_redis_queue_pool() -> None:
    if queue.pool is not None:
        logger.info("正在关闭 Redis 队列连接")
        await queue.pool.aclose()


# -------------- rate limit --------------
async def create_redis_rate_limit_pool() -> None:
    if settings.SKIP_INFRA_ON_START:
        logger.info("已跳过 Redis 限流连接（SKIP_INFRA_ON_START=true）")
        return
    logger.info("正在连接 Redis 限流: %s", settings.REDIS_RATE_LIMIT_URL)
    rate_limit.pool = redis.ConnectionPool.from_url(settings.REDIS_RATE_LIMIT_URL)
    rate_limit.client = redis.Redis.from_pool(rate_limit.pool)  # type: ignore
    await rate_limit.client.ping()
    logger.info("Redis 限流连接成功: %s", settings.REDIS_RATE_LIMIT_URL)


async def close_redis_rate_limit_pool() -> None:
    if rate_limit.client is not None:
        logger.info("正在关闭 Redis 限流连接")
        await rate_limit.client.aclose()


# -------------- application --------------
async def set_threadpool_tokens(number_of_tokens: int = 100) -> None:
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = number_of_tokens


def lifespan_factory(
    settings: Settings,
    create_tables_on_start: bool = True,
) -> Callable[[FastAPI], _AsyncGeneratorContextManager[Any]]:
    """Factory to create a lifespan async context manager for a FastAPI app."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator:
        await set_threadpool_tokens()
        logging.getLogger().setLevel(settings.LOG_LEVEL.upper())
        logger.info("应用启动，环境=%s，日志级别=%s", settings.ENVIRONMENT, settings.LOG_LEVEL)

        if (
            isinstance(settings, DatabaseSettings)
            and create_tables_on_start
            and not (isinstance(settings, SkipInfraSettings) and settings.SKIP_INFRA_ON_START)
        ):
            await create_tables()

        if isinstance(settings, RedisCacheSettings):
            await create_redis_cache_pool()

        if isinstance(settings, RedisQueueSettings):
            await create_redis_queue_pool()

        if isinstance(settings, RedisRateLimiterSettings):
            await create_redis_rate_limit_pool()

        yield

        if isinstance(settings, RedisCacheSettings):
            await close_redis_cache_pool()

        if isinstance(settings, RedisQueueSettings):
            await close_redis_queue_pool()

        if isinstance(settings, RedisRateLimiterSettings):
            await close_redis_rate_limit_pool()

    return lifespan


# -------------- application --------------
def create_application(
    router: APIRouter,
    settings: Settings,
    create_tables_on_start: bool = True,
    **kwargs: Any,
) -> FastAPI:
    """Creates and configures a FastAPI application based on the provided settings.

    This function initializes a FastAPI application and configures it with various settings
    and handlers based on the type of the `settings` object provided.

    Parameters
    ----------
    router : APIRouter
        The APIRouter object containing the routes to be included in the FastAPI application.

    settings
        An instance representing the settings for configuring the FastAPI application.
        It determines the configuration applied:

        - AppSettings: Configures basic app metadata like name, description, contact, and license info.
        - DatabaseSettings: Adds event handlers for initializing database tables during startup.
        - RedisCacheSettings: Sets up event handlers for creating and closing a Redis cache pool.
        - ClientSideCacheSettings: Integrates middleware for client-side caching.
        - RedisQueueSettings: Sets up event handlers for creating and closing a Redis queue pool.
        - RedisRateLimiterSettings: Sets up event handlers for creating and closing a Redis rate limiter pool.
        - EnvironmentSettings: Conditionally sets documentation URLs and integrates custom routes for API documentation
          based on the environment type.

    create_tables_on_start : bool
        A flag to indicate whether to create database tables on application startup.
        Defaults to True.

    **kwargs
        Additional keyword arguments passed directly to the FastAPI constructor.

    Returns
    -------
    FastAPI
        A fully configured FastAPI application instance.

    The function configures the FastAPI application with different features and behaviors
    based on the provided settings. It includes setting up database connections, Redis pools
    for caching, queue, and rate limiting, client-side caching, and customizing the API documentation
    based on the environment settings.
    """
    # --- before creating application ---
    if isinstance(settings, AppSettings):
        to_update = {
            "title": settings.APP_NAME,
            "description": settings.APP_DESCRIPTION,
            "contact": {"name": settings.CONTACT_NAME, "email": settings.CONTACT_EMAIL},
            "license_info": {"name": settings.LICENSE_NAME},
        }
        kwargs.update(to_update)

    if isinstance(settings, EnvironmentSettings):
        kwargs.update({"docs_url": None, "redoc_url": None, "openapi_url": None})

    lifespan = lifespan_factory(settings, create_tables_on_start=create_tables_on_start)

    application = FastAPI(lifespan=lifespan, **kwargs)
    application.include_router(router)

    if isinstance(settings, ClientSideCacheSettings):
        application.add_middleware(ClientCacheMiddleware, max_age=settings.CLIENT_CACHE_MAX_AGE)  # type: ignore[arg-type]

    if isinstance(settings, EnvironmentSettings):
        if settings.ENVIRONMENT != EnvironmentOption.PRODUCTION:
            docs_router = APIRouter()

            @docs_router.get("/docs", include_in_schema=False)
            async def get_swagger_documentation() -> fastapi.responses.HTMLResponse:
                return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")

            @docs_router.get("/redoc", include_in_schema=False)
            async def get_redoc_documentation() -> fastapi.responses.HTMLResponse:
                return get_redoc_html(openapi_url="/openapi.json", title="docs")

            @docs_router.get("/openapi.json", include_in_schema=False)
            async def openapi() -> dict[str, Any]:
                out: dict = get_openapi(title=application.title, version=application.version, routes=application.routes)
                return out

            application.include_router(docs_router)

        return application

    return application
