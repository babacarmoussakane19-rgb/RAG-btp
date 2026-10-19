"""Étape 3 : embeddings denses (sentence-transformers). Import paresseux : inutile pour BM25 seul."""
from __future__ import annotations

import numpy as np


class Embedder:
    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        passage_prefix: str = "",
        batch_size: int = 32,
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def _encode(self, texts: list[str], prefix: str, progress: bool) -> np.ndarray:
        vecs = self.model.encode(
            [prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,  # produit scalaire == similarité cosinus
            show_progress_bar=progress,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.passage_prefix, progress=True)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix, progress=False)

    @classmethod
    def from_config(cls, cfg: dict) -> "Embedder":
        e = cfg["embedding"]
        return cls(e["model"], e["query_prefix"], e["passage_prefix"], e["batch_size"])
