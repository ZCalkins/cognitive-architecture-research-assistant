"""Tests for paper ingestion (arxiv source + dedup/embed/store pipeline).

Hermetic: arxiv is mocked; the KB and embedder are fakes. Serves (A) — the
ingestion feed is the environment engagement differentiates within.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from src.ingestion import ArxivSource, IngestionPipeline
from src.kb import FakeKBConnector, Paper


def _fake_arxiv_result(arxiv_id, title, published):
    return SimpleNamespace(
        get_short_id=lambda: arxiv_id,
        title=title,
        authors=[SimpleNamespace(name="Jane Doe")],
        summary="An abstract.",
        categories=["cs.AI"],
        published=published,
        pdf_url=f"http://arxiv.org/pdf/{arxiv_id}",
        primary_category="cs.AI",
    )


def test_arxiv_source_translates_result_to_paper():
    """Serves (A): an arxiv result maps cleanly to a canonical Paper."""
    source = ArxivSource(categories=["cs.AI"])
    paper = source._to_paper(_fake_arxiv_result("2401.1", "  Title  ", datetime(2024, 1, 2)))
    assert paper.paper_id == "2401.1"
    assert paper.arxiv_id == "2401.1"
    assert paper.title == "Title"
    assert paper.authors == ["Jane Doe"]
    assert paper.categories == ["cs.AI"]
    assert paper.submitted_date == datetime(2024, 1, 2)


def test_arxiv_source_filters_by_since_date():
    """Serves (A): only papers at/after `since` are yielded."""
    source = ArxivSource(categories=["cs.AI"])
    source._arxiv = MagicMock()
    source._client = MagicMock()
    source._client.results.return_value = [
        _fake_arxiv_result("old", "Old", datetime(2020, 1, 1)),
        _fake_arxiv_result("new", "New", datetime(2024, 6, 1)),
    ]
    papers = list(source.fetch_recent(since=datetime(2024, 1, 1)))
    assert [paper.paper_id for paper in papers] == ["new"]


def test_ingestion_pipeline_dedups_by_paper_id():
    """Serves (A): a duplicate paper id is skipped, not re-embedded."""
    embedder = MagicMock()
    embedder.cached_embed_paper.return_value = torch.ones(8)
    source = MagicMock()
    source.source_name = "s"
    paper = Paper("a", "A", [], "abs")
    source.fetch_recent.return_value = [paper, paper]
    report = IngestionPipeline([source], FakeKBConnector(), embedder).run()
    assert report.papers_added == 1
    assert report.papers_skipped_duplicates == 1


def test_ingestion_pipeline_returns_report_with_counts():
    """Serves (D): the run returns an inspectable structured report."""
    embedder = MagicMock()
    embedder.cached_embed_paper.return_value = torch.ones(8)
    source = MagicMock()
    source.source_name = "arxiv"
    source.fetch_recent.return_value = [Paper("a", "A", [], "x"), Paper("b", "B", [], "y")]
    report = IngestionPipeline([source], FakeKBConnector(), embedder).run()
    assert report.papers_seen == 2
    assert report.papers_added == 2
    assert report.sources_run == ["arxiv"]
    assert report.duration_seconds >= 0.0


def test_ingestion_pipeline_uses_cached_embeddings():
    """Serves runtime model: ingestion goes through the embedding cache."""
    embedder = MagicMock()
    embedder.cached_embed_paper.return_value = torch.ones(8)
    source = MagicMock()
    source.source_name = "s"
    source.fetch_recent.return_value = [Paper("a", "A", [], "x")]
    IngestionPipeline([source], FakeKBConnector(), embedder).run()
    embedder.cached_embed_paper.assert_called_once()
    embedder.embed_paper.assert_not_called()


@pytest.mark.network
def test_arxiv_real_call_smoke():
    """Serves (A): real arxiv fetch (gated by the network marker)."""
    source = ArxivSource(categories=["cs.AI"], max_results_per_run=1)
    papers = list(source.fetch_recent(max_results=1))
    assert isinstance(papers, list)
