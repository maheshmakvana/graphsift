"""Schema versions for memory models (MemoryFact, SessionInfo) with migration paths.

Version history:
  v1 — Original: MemoryFact with basic fields, SessionInfo with summary.
  v2 — Current: adds schema_version, linked_symbols as tuple list, extended context.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# MemoryFact schemas
# ---------------------------------------------------------------------------


class MemoryFactV1(BaseModel):
    """Original MemoryFact — basic fields, no schema_version."""

    fact_id: str
    content: str
    session_id: str
    linked_symbols: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    valid_until: datetime | None = None
    invalid_at: datetime | None = None
    access_count: int = 0
    last_accessed: datetime

    def is_expired(self) -> bool:
        """Return True when the fact's TTL has elapsed."""
        if self.valid_until is None:
            return False
        return datetime.now(timezone.utc) > self.valid_until

    def is_deleted(self) -> bool:
        """Return True when the fact has been soft-deleted."""
        return self.invalid_at is not None


class MemoryFactV2(BaseModel):
    """Current MemoryFact — adds schema_version and weighted symbol links."""

    fact_id: str
    content: str
    session_id: str
    linked_symbols: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of {symbol: str, weight: float} dicts",
    )
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    valid_until: datetime | None = None
    invalid_at: datetime | None = None
    access_count: int = 0
    last_accessed: datetime
    schema_version: int = Field(default=2, ge=1)

    def is_expired(self) -> bool:
        if self.valid_until is None:
            return False
        return datetime.now(timezone.utc) > self.valid_until

    def is_deleted(self) -> bool:
        return self.invalid_at is not None


# ---------------------------------------------------------------------------
# SessionInfo schemas
# ---------------------------------------------------------------------------


class SessionInfoV1(BaseModel):
    """Original SessionInfo — basic summary fields."""

    session_id: str
    fact_count: int
    created_at: datetime
    last_accessed: datetime
    summary: str = ""


class SessionInfoV2(BaseModel):
    """Current SessionInfo — adds schema_version and session metadata."""

    session_id: str
    fact_count: int
    created_at: datetime
    last_accessed: datetime
    summary: str = ""
    token_estimate: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = Field(default=2, ge=1)


# ---------------------------------------------------------------------------
# Migration helpers
# ---------------------------------------------------------------------------


def migrate_memory_fact_v1_to_v2(data: dict) -> dict:
    """Migrate a MemoryFactV1 dict to MemoryFactV2 format.

    Converts plain symbol strings to {symbol, weight} dicts.
    """
    result = dict(data)
    symbols: list[str] = result.get("linked_symbols", [])
    result["linked_symbols"] = [
        {"symbol": s, "weight": 1.0} for s in symbols
    ]
    result["schema_version"] = 2
    return result


def migrate_session_info_v1_to_v2(data: dict) -> dict:
    """Migrate a SessionInfoV1 dict to SessionInfoV2 format."""
    result = dict(data)
    result.setdefault("token_estimate", 0)
    result.setdefault("metadata", {})
    result["schema_version"] = 2
    return result
