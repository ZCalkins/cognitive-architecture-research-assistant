"""Paper sources — the system's environment for the v0 task surface.

Strange-loop role
-----------------
Paper sources are the system's environment. The (A) personalization claim
depends on the source being broad enough that engagement-signal differentiation
across papers is meaningful — a narrow feed would make any apparent
personalization an artifact of the feed, not the system.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime
from typing import Any, Protocol

from dateutil import parser as date_parser

from src.kb.types import Paper

logger = logging.getLogger(__name__)


class PaperSource(Protocol):
    """A source of papers (the system's environment)."""

    source_name: str

    def fetch_recent(
        self, since: datetime | None = None, max_results: int = 50
    ) -> Iterator[Paper]: ...

    def fetch_by_id(self, source_id: str) -> Paper | None: ...


def _as_naive(value: datetime) -> datetime:
    """Drop tzinfo for comparison; arxiv publishes tz-aware UTC datetimes."""
    return value.replace(tzinfo=None) if value.tzinfo is not None else value


class ArxivSource:
    """Ingests papers from the arxiv API via the ``arxiv`` package."""

    source_name = "arxiv"

    def __init__(self, categories: list[str], max_results_per_run: int = 50) -> None:
        self.categories = categories
        self.max_results_per_run = max_results_per_run
        import arxiv

        self._arxiv = arxiv
        self._client = arxiv.Client()

    def _query(self) -> str:
        return " OR ".join(f"cat:{category}" for category in self.categories)

    def _to_paper(self, result: Any) -> Paper:
        arxiv_id = result.get_short_id()
        published = result.published
        if isinstance(published, datetime):
            submitted = published
        elif published:
            submitted = date_parser.parse(str(published))
        else:
            submitted = None
        return Paper(
            paper_id=arxiv_id,
            title=result.title.strip(),
            authors=[author.name for author in result.authors],
            abstract=result.summary.strip(),
            arxiv_id=arxiv_id,
            categories=list(result.categories),
            submitted_date=submitted,
            pdf_url=result.pdf_url,
            metadata={"primary_category": result.primary_category},
        )

    def fetch_recent(
        self, since: datetime | None = None, max_results: int = 50
    ) -> Iterator[Paper]:
        search = self._arxiv.Search(
            query=self._query(),
            max_results=max_results,
            sort_by=self._arxiv.SortCriterion.SubmittedDate,
            sort_order=self._arxiv.SortOrder.Descending,
        )
        cutoff = _as_naive(since) if since is not None else None
        for result in self._client.results(search):
            paper = self._to_paper(result)
            if (
                cutoff is not None
                and paper.submitted_date is not None
                and _as_naive(paper.submitted_date) < cutoff
            ):
                continue
            logger.info("arxiv fetched %s: %s", paper.paper_id, paper.title)
            yield paper

    def fetch_by_id(self, source_id: str) -> Paper | None:
        search = self._arxiv.Search(id_list=[source_id])
        for result in self._client.results(search):
            return self._to_paper(result)
        return None
