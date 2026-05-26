"""Tests for the local KB (semantic-memory track).

Hermetic: the embedding model is faked; FAISS/SQLite run in a tempdir. Serves
(A) personalization (the KB is the user's corpus) and the (D) falsifiability
discipline (tests touch no network and no model download).
"""

import hashlib
from datetime import datetime

import numpy as np
import torch

from src.kb import EmbeddingService, FakeKBConnector, LocalKBConnector, Paper, PaperRef


class _FakeModel:
    """Deterministic stand-in for a SentenceTransformer."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls = 0

    def encode(self, texts, batch_size: int = 32, convert_to_numpy: bool = True):
        self.calls += 1
        rows = []
        for text in texts:
            seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)
            rows.append(np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32))
        return np.stack(rows)


def _fake_service(tmp_path=None, dim: int = 8) -> EmbeddingService:
    service = EmbeddingService(cache_dir=tmp_path)
    service._model = _FakeModel(dim=dim)
    return service


def test_paper_dataclass_construction():
    """Serves (A): canonical paper representation with sane defaults."""
    paper = Paper(paper_id="1", title="T", authors=["A"], abstract="abs")
    assert paper.paper_id == "1"
    assert paper.categories == []
    assert paper.metadata == {}


def test_paper_ref_lightweight():
    """Serves (A): a lightweight result type defaults score to 0."""
    assert PaperRef(paper_id="1", title="T").score == 0.0


def test_embedding_service_embeds_single_text():
    """Serves (A): single-text embedding has the model dimension."""
    assert _fake_service(dim=8).embed_text("hello").shape == (8,)


def test_embedding_service_embeds_paper_combines_title_and_abstract():
    """Serves (A): paper embedding combines title and abstract."""
    service = _fake_service(dim=8)
    captured: dict = {}
    inner = service._model.encode

    def spy(texts, **kwargs):
        captured["texts"] = texts
        return inner(texts, **kwargs)

    service._model.encode = spy
    service.embed_paper(Paper("1", "MyTitle", [], "MyAbstract"))
    assert "MyTitle" in captured["texts"][0]
    assert "MyAbstract" in captured["texts"][0]


def test_embedding_service_cache_hit(tmp_path):
    """Serves runtime model: a cached paper is not re-embedded."""
    service = _fake_service(tmp_path, dim=8)
    paper = Paper("P1", "T", [], "A")
    first = service.cached_embed_paper(paper)
    calls = service._model.calls
    second = service.cached_embed_paper(paper)
    assert service._model.calls == calls
    assert torch.allclose(first, second)


def test_embedding_service_cache_invalidates_on_text_change():
    """Serves runtime model: changed content invalidates the cache."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        service = _fake_service(tmp, dim=8)
        service.cached_embed_paper(Paper("P1", "T", [], "A"))
        service.cached_embed_paper(Paper("P1", "T", [], "A CHANGED"))
        assert service._model.calls == 2


def test_fake_kb_round_trip_paper():
    """Serves (D): the hermetic KB round-trips a paper."""
    kb = FakeKBConnector()
    kb.add_paper(Paper("1", "T", ["A"], "abs"), torch.ones(8))
    assert kb.has_paper("1")
    assert kb.get_paper("1").title == "T"
    assert kb.count() == 1


def test_fake_kb_search_by_embedding_returns_top_k_sorted():
    """Serves (A): similarity search returns top-k, descending."""
    kb = FakeKBConnector()
    kb.add_paper(Paper("a", "A", [], "x"), torch.tensor([1.0, 0.0]))
    kb.add_paper(Paper("b", "B", [], "x"), torch.tensor([0.0, 1.0]))
    kb.add_paper(Paper("c", "C", [], "x"), torch.tensor([0.9, 0.1]))
    refs = kb.search_by_embedding(torch.tensor([1.0, 0.0]), top_k=2)
    assert len(refs) == 2
    assert refs[0].paper_id == "a"
    assert refs[0].score >= refs[1].score


def test_fake_kb_nearest_neighbors_excludes_self():
    """Serves (A): nearest-neighbors excludes the query paper."""
    kb = FakeKBConnector()
    kb.add_paper(Paper("a", "A", [], "x"), torch.tensor([1.0, 0.0]))
    kb.add_paper(Paper("b", "B", [], "x"), torch.tensor([0.9, 0.1]))
    refs = kb.nearest_neighbors("a", top_k=5)
    assert all(ref.paper_id != "a" for ref in refs)
    assert refs[0].paper_id == "b"


def test_fake_kb_tag_paper_and_retrieve():
    """Serves (A): tags (engagement) attach to papers and are queryable."""
    kb = FakeKBConnector()
    kb.add_paper(Paper("a", "A", [], "x"), torch.ones(2))
    kb.tag_paper("a", "read")
    refs = kb.papers_with_tag("read")
    assert len(refs) == 1
    assert refs[0].paper_id == "a"


def test_local_kb_round_trip_with_tempdir(tmp_path):
    """Serves (A): the on-disk KB round-trips a paper with full fields."""
    service = _fake_service(dim=8)
    kb = LocalKBConnector(
        tmp_path / "papers.db", tmp_path / "p.faiss", tmp_path / "m.jsonl", service, embedding_dim=8
    )
    paper = Paper(
        "a", "A", ["Auth"], "abs", arxiv_id="2401.1", categories=["cs.AI"],
        submitted_date=datetime(2024, 1, 1),
    )
    kb.add_paper(paper, torch.ones(8))
    got = kb.get_paper("a")
    assert got.title == "A"
    assert got.arxiv_id == "2401.1"
    assert got.categories == ["cs.AI"]
    assert kb.count() == 1
    assert kb.search_by_embedding(torch.ones(8), top_k=1)[0].paper_id == "a"
    kb.close()


def test_local_kb_persistence_across_instances(tmp_path):
    """Serves Phase 5 criterion 5: KB state survives a cold restart."""
    service = _fake_service(dim=8)
    paths = (tmp_path / "papers.db", tmp_path / "p.faiss", tmp_path / "m.jsonl")
    kb = LocalKBConnector(*paths, service, embedding_dim=8)
    kb.add_paper(Paper("a", "A", [], "abs"), torch.ones(8))
    kb.close()
    reopened = LocalKBConnector(*paths, service, embedding_dim=8)
    assert reopened.count() == 1
    assert reopened.has_paper("a")
    assert reopened.search_by_embedding(torch.ones(8), top_k=1)[0].paper_id == "a"
    reopened.close()


def test_local_kb_faiss_index_rebuilds_on_load(tmp_path):
    """Serves Phase 5 criterion 5: the FAISS index reloads from disk intact."""
    service = _fake_service(dim=8)
    paths = (tmp_path / "papers.db", tmp_path / "p.faiss", tmp_path / "m.jsonl")
    kb = LocalKBConnector(*paths, service, embedding_dim=8)
    kb.add_paper(Paper("a", "A", [], "abs"), torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0]))
    kb.add_paper(Paper("b", "B", [], "abs"), torch.tensor([0, 1.0, 0, 0, 0, 0, 0, 0]))
    kb.close()
    reopened = LocalKBConnector(*paths, service, embedding_dim=8)
    neighbors = reopened.nearest_neighbors("a", top_k=5)
    assert any(ref.paper_id == "b" for ref in neighbors)
    reopened.close()
