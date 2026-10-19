"""Étape 4 : index hybride (BM25 + vecteurs denses) persistant sur disque.

Choix volontairement simple : la similarité dense est un produit matriciel NumPy, largement
suffisant jusqu'à ~100 000 chunks. Pour passer à l'échelle, remplacer ``dense_scores`` par
FAISS, Qdrant ou pgvector sans toucher au reste du pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .textutils import tokenize


class Index:
    def __init__(self, chunks: list[dict], embeddings: np.ndarray | None = None, embedding_model: str | None = None):
        if embeddings is not None and len(embeddings) != len(chunks):
            raise ValueError("embeddings et chunks doivent avoir la même longueur")
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedding_model = embedding_model
        self._bm25 = BM25Okapi([tokenize(c["embed_text"]) for c in chunks]) if chunks else None

    # --- scores -----------------------------------------------------------------------------
    def bm25_scores(self, query: str) -> np.ndarray:
        if self._bm25 is None:
            return np.zeros(0)
        return np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)

    def dense_scores(self, query_vec: np.ndarray) -> np.ndarray:
        if self.embeddings is None:
            raise RuntimeError("Index sans embeddings : relancer `rag-btp build` sans --no-embed.")
        return self.embeddings @ query_vec

    def mask(self, filters: dict | None) -> np.ndarray:
        """Masque booléen selon des filtres de métadonnées, ex. {"statut": "en_vigueur"}."""
        m = np.ones(len(self.chunks), dtype=bool)
        for key, wanted in (filters or {}).items():
            allowed = wanted if isinstance(wanted, (list, tuple, set)) else [wanted]
            for i, c in enumerate(self.chunks):
                if m[i] and c.get(key) not in allowed:
                    m[i] = False
        return m

    # --- persistance ------------------------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with (d / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        if self.embeddings is not None:
            np.save(d / "embeddings.npy", self.embeddings)
        info = {"n_chunks": len(self.chunks), "embedding_model": self.embedding_model}
        (d / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "Index":
        d = Path(directory)
        if not (d / "chunks.jsonl").exists():
            raise FileNotFoundError(f"Aucun index dans {d} : lancer `rag-btp build` d'abord.")
        chunks = [json.loads(line) for line in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
        emb_path = d / "embeddings.npy"
        embeddings = np.load(emb_path) if emb_path.exists() else None
        info = json.loads((d / "info.json").read_text(encoding="utf-8")) if (d / "info.json").exists() else {}
        return cls(chunks, embeddings, info.get("embedding_model"))
