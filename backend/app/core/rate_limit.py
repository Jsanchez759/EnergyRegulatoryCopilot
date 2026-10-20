import time
from functools import lru_cache

from limits import RateLimitItem, parse_many
from limits.aio.strategies import MovingWindowRateLimiter
from limits.storage import storage_from_string

from app.core.config import settings


@lru_cache
def get_limiter() -> MovingWindowRateLimiter:
    uri = settings.RATE_LIMIT_STORAGE_URI
    if not uri.startswith("async+"):
        uri = f"async+{uri}"
    return MovingWindowRateLimiter(storage_from_string(uri))


@lru_cache
def parse_limits(value: str) -> list[RateLimitItem]:
    return parse_many(value)


async def consume(scope: str, key: str, limit_value: str) -> int | None:
    """Record one hit; return seconds until retry when any limit is exceeded."""
    limiter = get_limiter()
    for item in parse_limits(limit_value):
        if not await limiter.hit(item, scope, key):
            stats = await limiter.get_window_stats(item, scope, key)
            return max(1, int(stats.reset_time - time.time()) + 1)
    return None
