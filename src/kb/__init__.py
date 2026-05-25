"""Local KB — the semantic-memory track (ARCHITECTURE.md "Memory architecture").

v0 owns its own paper corpus locally (SQLite + FAISS + sentence-transformers).
No ResearchKB, no external embedding API. See CLAUDE.md stack constraints.
"""

from src.kb.connector import FakeKBConnector, KBConnector, LocalKBConnector
from src.kb.embedding_service import EmbeddingService
from src.kb.types import Paper, PaperRef

__all__ = [
    "EmbeddingService",
    "FakeKBConnector",
    "KBConnector",
    "LocalKBConnector",
    "Paper",
    "PaperRef",
]
