"""Paper ingestion — arxiv source + dedup/embed/store pipeline (Session 3).

See ARCHITECTURE.md "Task surface" (v0) and "Phase 1 deliverable 5".
"""

from src.ingestion.pipeline import IngestionPipeline, IngestionReport
from src.ingestion.sources import ArxivSource, PaperSource

__all__ = ["ArxivSource", "IngestionPipeline", "IngestionReport", "PaperSource"]
