"""Shared router dependencies and parameter types."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from fastapi import Path as PathParam

from backend.app.security import get_client_id
from schemas.common import SESSION_ID_PATTERN

SessionIdPath = Annotated[
    str,
    PathParam(pattern=SESSION_ID_PATTERN, description="Session id ([A-Za-z0-9_-]{1,64})."),
]
ClientId = Annotated[str, Depends(get_client_id)]
