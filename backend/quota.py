"""
quota.py
Redis-based user quota enforcement.

Replaces usage_db.json which had a race condition: two concurrent requests from
the same user would both read count=0, both increment to 1, and both write —
effectively ignoring the quota limit.

Fix: Redis INCR is atomic. The counter can never be double-incremented by
concurrent requests. EXPIRE sets a 24-hour TTL so quotas auto-reset daily
without any cron job or manual reset logic.
"""
import os
from redis_client import redis_client

MAX_PRS_PER_USER = 2
MAX_TOTAL_PRS = 50
QUOTA_TTL_SECONDS = 86400  # 24 hours

OWNER_USERNAME = "harshh06"  # always bypasses quota


def check_user_quota(username: str) -> bool:
    """
    Returns True if the user is within quota and increments their counter.
    Returns False if the user has exceeded their limit.

    Uses Redis INCR (atomic) to prevent race conditions between concurrent
    webhook deliveries from the same user.
    """
    # Owner is always allowed — never gets locked out of their own project
    if username == OWNER_USERNAME:
        return True

    # --- Global hard cap ---
    global_count = redis_client.incr("quota:GLOBAL_TOTAL")
    if global_count == 1:
        # First ever increment — set the TTL (24h reset)
        redis_client.expire("quota:GLOBAL_TOTAL", QUOTA_TTL_SECONDS)

    if global_count > MAX_TOTAL_PRS:
        # Undo the increment we just did — this user didn't actually get a slot
        redis_client.decr("quota:GLOBAL_TOTAL")
        return False

    # --- Per-user cap ---
    user_key = f"quota:user:{username}"
    user_count = redis_client.incr(user_key)
    if user_count == 1:
        redis_client.expire(user_key, QUOTA_TTL_SECONDS)

    if user_count > MAX_PRS_PER_USER:
        # Undo both increments
        redis_client.decr(user_key)
        redis_client.decr("quota:GLOBAL_TOTAL")
        return False

    return True
