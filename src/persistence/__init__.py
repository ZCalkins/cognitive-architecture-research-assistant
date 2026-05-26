"""SQLite-backed persistence (ARCHITECTURE.md "Runtime model").

SQLite for v0; Postgres if/when concurrency demands it. No SQLAlchemy
(CLAUDE.md stack constraints).
"""

from src.persistence.store import SQLiteStore, StoreEvent

__all__ = ["SQLiteStore", "StoreEvent"]
