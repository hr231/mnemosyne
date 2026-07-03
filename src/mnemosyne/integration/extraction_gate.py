from __future__ import annotations

import uuid

_SELECT_FLAG_SQL = """
    SELECT extraction_enabled
    FROM memory.user_settings
    WHERE user_id = $1
"""


async def is_extraction_enabled_for(pool, user_id: uuid.UUID) -> bool:
    """Return whether background extraction is enabled for ``user_id``.

    Reads the durable flag from ``memory.user_settings``. A missing row (or no
    pool) defaults to enabled, so existing users keep extracting until they
    explicitly opt out. Shared by the management service and the background
    pipeline so both honour the same source of truth.
    """
    if pool is None:
        return True
    async with pool.acquire() as conn:
        value = await conn.fetchval(_SELECT_FLAG_SQL, user_id)
    if value is None:
        return True
    return bool(value)
