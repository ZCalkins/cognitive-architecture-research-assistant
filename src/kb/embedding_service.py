"""Local paper embeddings via sentence-transformers (no API, no key, CPU at v0 scale).

Strange-loop role
-----------------
Embeddings are the geometry the KB's semantic-memory track is organized by.
They are computed locally so the v0 system runs entirely on the user's machine
— a precondition for the (A) personalization claim being about *this user's*
corpus and engagement, not a vendor's. The model is loaded lazily so importing
this module (and running the hermetic test suite) needs no model download.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from src.kb.types import Paper


class EmbeddingService:
    """Lazily-loaded sentence-transformers embedder with a file-based cache.

    Parameters
    ----------
    model_name:
        sentence-transformers model id.
    cache_dir:
        Directory for the per-paper embedding cache; ``None`` disables caching.
    device:
        Torch device. Defaults to CPU because the GPU is reserved for Ollama.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/mxbai-embed-large-v1",
        cache_dir: Path | str | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.device = device
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_text(self, text: str) -> Tensor:
        vector = self._get_model().encode([text], convert_to_numpy=True)[0]
        return torch.from_numpy(np.asarray(vector, dtype=np.float32))

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> Tensor:
        vectors = self._get_model().encode(
            texts, batch_size=batch_size, convert_to_numpy=True
        )
        return torch.from_numpy(np.asarray(vectors, dtype=np.float32))

    @staticmethod
    def _paper_text(paper: Paper) -> str:
        return f"{paper.title}\n\n{paper.abstract}"

    def embed_paper(self, paper: Paper) -> Tensor:
        return self.embed_text(self._paper_text(paper))

    def cached_embed_paper(self, paper: Paper) -> Tensor:
        """Embed a paper, caching by paper_id + a content hash.

        The SHA-1 suffix over (title + abstract) invalidates the cache entry
        whenever the paper's content changes.
        """
        if self.cache_dir is None:
            return self.embed_paper(paper)
        digest = hashlib.sha1(self._paper_text(paper).encode("utf-8")).hexdigest()[:8]
        safe_id = "".join(c if c.isalnum() else "_" for c in paper.paper_id)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{safe_id}_{digest}.npy"
        if cache_path.exists():
            return torch.from_numpy(np.load(cache_path))
        embedding = self.embed_paper(paper)
        np.save(cache_path, embedding.detach().cpu().numpy())
        return embedding
