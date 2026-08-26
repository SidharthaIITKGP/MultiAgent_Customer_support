"""
auth.py — shared-secret API key auth for /api/* routes.

Not OAuth/JWT: there's no real user-identity system in this app (user_id is a
free-form label, not a login), so building one would invent a requirement.
A shared-secret header is the proportionate way to keep the backend from being
open to the whole internet.
"""

from __future__ import annotations

import os

from fastapi import Header, HTTPException, status


def _load_api_keys() -> set[str]:
    raw = os.getenv("API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


def require_api_key(x_api_key: str = Header(default="", alias="X-API-Key")) -> str:
    valid_keys = _load_api_keys()
    if not valid_keys or x_api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header.",
        )
    return x_api_key
