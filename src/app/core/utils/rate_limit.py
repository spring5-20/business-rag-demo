from datetime import UTC, datetime

from redis.asyncio import ConnectionPool, Redis
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.logger import logging

logger = logging.getLogger(__name__)

pool: ConnectionPool | None = None
client: Redis | None = None


def sanitize_path(path: str) -> str:
    return path.strip("/").replace("/", "_")


async def is_rate_limited(db: AsyncSession, user_id: int, path: str, limit: int, period: int) -> bool:
    if client is None:
        logger.error("Redis client is not initialized.")
        raise Exception("Redis client is not initialized.")

    current_timestamp = int(datetime.now(UTC).timestamp())
    window_start = current_timestamp - (current_timestamp % period)

    sanitized_path = sanitize_path(path)
    key = f"ratelimit:{user_id}:{sanitized_path}:{window_start}"

    try:
        current_count = await client.incr(key)
        if current_count == 1:
            await client.expire(key, period)

        if current_count > limit:
            logger.warning("触发限流: user=%s path=%s count=%s limit=%s", user_id, path, current_count, limit)
            return True
        logger.debug("限流计数: user=%s path=%s count=%s limit=%s", user_id, path, current_count, limit)

    except Exception as e:
        logger.exception(f"Error checking rate limit for user {user_id} on path {path}: {e}")
        raise e

    return False
