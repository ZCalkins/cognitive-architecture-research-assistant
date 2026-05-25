"""Ingestion pipeline — fetch, dedup, embed, store.

Strange-loop role
-----------------
The pipeline is the boundary where the environment (paper sources) enters the
system's semantic memory. Dedup + the embedding cache keep daily batches cheap
so the system can run unattended (the R2 runtime model).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ingestion.sources import PaperSource
    from src.kb.connector import KBConnector
    from src.kb.embedding_service import EmbeddingService


@dataclass
class IngestionReport:
    """Summary of one ingestion run."""

    papers_seen: int = 0
    papers_added: int = 0
    papers_skipped_duplicates: int = 0
    sources_run: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class IngestionPipeline:
    """Runs sources, dedups against the KB, embeds, and stores new papers."""

    def __init__(
        self,
        sources: list[PaperSource],
        kb: KBConnector,
        embedding_service: EmbeddingService,
    ) -> None:
        self.sources = sources
        self.kb = kb
        self.embedding_service = embedding_service

    def run(self, since: datetime | None = None) -> IngestionReport:
        start = time.time()
        report = IngestionReport()
        for source in self.sources:
            report.sources_run.append(source.source_name)
            for paper in source.fetch_recent(since=since):
                report.papers_seen += 1
                if self.kb.has_paper(paper.paper_id):
                    report.papers_skipped_duplicates += 1
                    continue
                embedding = self.embedding_service.cached_embed_paper(paper)
                self.kb.add_paper(paper, embedding)
                report.papers_added += 1
        report.duration_seconds = time.time() - start
        return report
