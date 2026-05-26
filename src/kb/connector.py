"""KB connectors — the semantic-memory track of the architecture.

Strange-loop role
-----------------
This is the semantic-memory track (one of the five memory tracks). v0 keeps it
strictly local: strange-loop owns its own paper corpus. A future ResearchKB
bridge (Phase 1.5+) would be a second ``KBConnector`` implementation, never a
replacement. Tests use ``FakeKBConnector`` to stay hermetic per the (D)
falsifiability discipline.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from torch import Tensor

from src.kb.types import Paper, PaperRef

if TYPE_CHECKING:
    from src.kb.embedding_service import EmbeddingService


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return (vector / norm).astype(np.float32)


class KBConnector(Protocol):
    """The semantic-memory interface the rest of the system depends on."""

    def has_paper(self, paper_id: str) -> bool: ...
    def get_paper(self, paper_id: str) -> Paper | None: ...
    def add_paper(self, paper: Paper, embedding: Tensor) -> None: ...
    def search_by_text(self, query: str, top_k: int = 10) -> list[PaperRef]: ...
    def search_by_embedding(self, embedding: Tensor, top_k: int = 10) -> list[PaperRef]: ...
    def nearest_neighbors(self, paper_id: str, top_k: int = 10) -> list[PaperRef]: ...
    def all_papers(self, limit: int | None = None) -> list[Paper]: ...
    def tag_paper(self, paper_id: str, tag: str, provenance: dict | None = None) -> None: ...
    def papers_with_tag(self, tag: str) -> list[PaperRef]: ...
    def count(self) -> int: ...


_PAPERS_DDL = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT,
    authors_json TEXT,
    abstract TEXT,
    arxiv_id TEXT,
    semantic_scholar_id TEXT,
    categories_json TEXT,
    submitted_date TEXT,
    pdf_url TEXT,
    metadata_json TEXT,
    embedding_index INTEGER,
    created_at REAL
)
"""

_TAGS_DDL = """
CREATE TABLE IF NOT EXISTS paper_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT,
    tag TEXT,
    provenance_json TEXT,
    ts REAL
)
"""


class LocalKBConnector:
    """Local KB backed by SQLite (metadata) + FAISS (similarity search).

    The FAISS ``IndexFlatIP`` over L2-normalized vectors yields cosine
    similarity. A JSONL metadata file maps FAISS positions to paper ids and is
    saved alongside the index.
    """

    def __init__(
        self,
        db_path: Path | str,
        index_path: Path | str,
        metadata_path: Path | str,
        embedding_service: EmbeddingService,
        embedding_dim: int = 1024,
    ) -> None:
        self.db_path = str(db_path)
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embedding_service = embedding_service
        self.embedding_dim = embedding_dim

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(_PAPERS_DDL)
        self.conn.execute(_TAGS_DDL)
        self.conn.commit()

        import faiss

        self._faiss = faiss
        self._positions: list[str] = []
        self._load_index()

    def _load_index(self) -> None:
        if self.index_path.exists() and self.metadata_path.exists():
            self._index = self._faiss.read_index(str(self.index_path))
            with open(self.metadata_path, encoding="utf-8") as handle:
                self._positions = [
                    json.loads(line)["paper_id"] for line in handle if line.strip()
                ]
        else:
            self._index = self._faiss.IndexFlatIP(self.embedding_dim)
            self._positions = []

    def _save_index(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(self.index_path))
        with open(self.metadata_path, "w", encoding="utf-8") as handle:
            for paper_id in self._positions:
                handle.write(json.dumps({"paper_id": paper_id}) + "\n")

    @staticmethod
    def _row_to_paper(row: sqlite3.Row) -> Paper:
        submitted = row["submitted_date"]
        return Paper(
            paper_id=row["paper_id"],
            title=row["title"],
            authors=json.loads(row["authors_json"]),
            abstract=row["abstract"],
            arxiv_id=row["arxiv_id"],
            semantic_scholar_id=row["semantic_scholar_id"],
            categories=json.loads(row["categories_json"]),
            submitted_date=datetime.fromisoformat(submitted) if submitted else None,
            pdf_url=row["pdf_url"],
            metadata=json.loads(row["metadata_json"]),
        )

    def has_paper(self, paper_id: str) -> bool:
        return (
            self.conn.execute(
                "SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,)
            ).fetchone()
            is not None
        )

    def get_paper(self, paper_id: str) -> Paper | None:
        row = self.conn.execute(
            "SELECT * FROM papers WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        return self._row_to_paper(row) if row is not None else None

    def add_paper(self, paper: Paper, embedding: Tensor) -> None:
        if self.has_paper(paper.paper_id):
            return
        vector = _l2_normalize(embedding.detach().cpu().numpy())
        position = self._index.ntotal
        self._index.add(vector.reshape(1, -1))
        self._positions.append(paper.paper_id)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO papers (
                paper_id, title, authors_json, abstract, arxiv_id,
                semantic_scholar_id, categories_json, submitted_date, pdf_url,
                metadata_json, embedding_index, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                paper.paper_id,
                paper.title,
                json.dumps(paper.authors),
                paper.abstract,
                paper.arxiv_id,
                paper.semantic_scholar_id,
                json.dumps(paper.categories),
                paper.submitted_date.isoformat() if paper.submitted_date else None,
                paper.pdf_url,
                json.dumps(paper.metadata, default=str),
                position,
                time.time(),
            ),
        )
        self.conn.commit()
        self._save_index()

    def search_by_embedding(self, embedding: Tensor, top_k: int = 10) -> list[PaperRef]:
        if self._index.ntotal == 0:
            return []
        vector = _l2_normalize(embedding.detach().cpu().numpy()).reshape(1, -1)
        scores, indices = self._index.search(vector, min(top_k, self._index.ntotal))
        return self._refs_from_search(scores[0], indices[0], exclude=None, limit=top_k)

    def search_by_text(self, query: str, top_k: int = 10) -> list[PaperRef]:
        return self.search_by_embedding(self.embedding_service.embed_text(query), top_k)

    def nearest_neighbors(self, paper_id: str, top_k: int = 10) -> list[PaperRef]:
        row = self.conn.execute(
            "SELECT embedding_index FROM papers WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        if row is None or self._index.ntotal == 0:
            return []
        vector = self._index.reconstruct(int(row["embedding_index"])).reshape(1, -1)
        scores, indices = self._index.search(
            vector, min(top_k + 1, self._index.ntotal)
        )
        return self._refs_from_search(scores[0], indices[0], exclude=paper_id, limit=top_k)

    def _refs_from_search(self, scores, indices, exclude: str | None, limit: int) -> list[PaperRef]:
        refs: list[PaperRef] = []
        for score, index in zip(scores, indices, strict=False):
            if index < 0:
                continue
            paper_id = self._positions[index]
            if paper_id == exclude:
                continue
            paper = self.get_paper(paper_id)
            refs.append(PaperRef(paper_id, paper.title if paper else paper_id, float(score)))
            if len(refs) >= limit:
                break
        return refs

    def all_papers(self, limit: int | None = None) -> list[Paper]:
        sql = "SELECT * FROM papers ORDER BY created_at ASC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return [self._row_to_paper(row) for row in self.conn.execute(sql).fetchall()]

    def tag_paper(self, paper_id: str, tag: str, provenance: dict | None = None) -> None:
        self.conn.execute(
            "INSERT INTO paper_tags (paper_id, tag, provenance_json, ts) VALUES (?, ?, ?, ?)",
            (paper_id, tag, json.dumps(provenance or {}, default=str), time.time()),
        )
        self.conn.commit()

    def papers_with_tag(self, tag: str) -> list[PaperRef]:
        rows = self.conn.execute(
            "SELECT DISTINCT paper_id FROM paper_tags WHERE tag = ?", (tag,)
        ).fetchall()
        refs: list[PaperRef] = []
        for row in rows:
            paper = self.get_paper(row["paper_id"])
            if paper is not None:
                refs.append(PaperRef(paper.paper_id, paper.title))
        return refs

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) AS c FROM papers").fetchone()["c"]

    def close(self) -> None:
        self._save_index()
        self.conn.close()


class FakeKBConnector:
    """In-memory KB for hermetic tests. Cosine similarity via normalized dot."""

    def __init__(self) -> None:
        self._papers: dict[str, Paper] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._tags: dict[str, list[tuple[str, dict]]] = {}

    def has_paper(self, paper_id: str) -> bool:
        return paper_id in self._papers

    def get_paper(self, paper_id: str) -> Paper | None:
        return self._papers.get(paper_id)

    def add_paper(self, paper: Paper, embedding: Tensor) -> None:
        self._papers[paper.paper_id] = paper
        self._embeddings[paper.paper_id] = _l2_normalize(embedding.detach().cpu().numpy())

    def search_by_text(self, query: str, top_k: int = 10) -> list[PaperRef]:
        return []

    def search_by_embedding(self, embedding: Tensor, top_k: int = 10) -> list[PaperRef]:
        if not self._embeddings:
            return []
        query = _l2_normalize(embedding.detach().cpu().numpy())
        scored = [
            (paper_id, float(np.dot(query, vector)))
            for paper_id, vector in self._embeddings.items()
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            PaperRef(paper_id, self._papers[paper_id].title, score)
            for paper_id, score in scored[:top_k]
        ]

    def nearest_neighbors(self, paper_id: str, top_k: int = 10) -> list[PaperRef]:
        if paper_id not in self._embeddings:
            return []
        query = self._embeddings[paper_id]
        scored = [
            (other_id, float(np.dot(query, vector)))
            for other_id, vector in self._embeddings.items()
            if other_id != paper_id
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            PaperRef(other_id, self._papers[other_id].title, score)
            for other_id, score in scored[:top_k]
        ]

    def all_papers(self, limit: int | None = None) -> list[Paper]:
        papers = list(self._papers.values())
        return papers[:limit] if limit is not None else papers

    def tag_paper(self, paper_id: str, tag: str, provenance: dict | None = None) -> None:
        self._tags.setdefault(paper_id, []).append((tag, provenance or {}))

    def papers_with_tag(self, tag: str) -> list[PaperRef]:
        return [
            PaperRef(paper_id, self._papers[paper_id].title)
            for paper_id, tags in self._tags.items()
            if paper_id in self._papers and any(existing == tag for existing, _ in tags)
        ]

    def count(self) -> int:
        return len(self._papers)

    def close(self) -> None:
        pass
