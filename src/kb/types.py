"""Canonical paper representations — the substrate the v0 task surface operates on.

Strange-loop role
-----------------
Paper representations are what the v0 paper-triage task surface operates on. In
Phase 2+ an individual paper may itself become a Schema; in Phase 4 papers are
indexed by self-narrative memory. Session 3 establishes the canonical
representation that ingestion, the KB, and the triage surface all share.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class PaperRef:
    """Lightweight reference to a paper (a search/result hit)."""

    paper_id: str
    title: str
    score: float = 0.0


@dataclass
class Paper:
    """A canonical paper representation."""

    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    categories: list[str] = field(default_factory=list)
    submitted_date: datetime | None = None
    pdf_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
