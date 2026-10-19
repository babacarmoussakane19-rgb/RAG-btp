"""Étape 5 : retrieval BM25 / dense / hybride (Reciprocal Rank Fusion) + filtres + reranking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .index import Index

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


@dataclass
class Hit:
    chunk: dict
    score: float
    rank: int  # 1 = meilleur


def _top(scores: np.ndarray, mask: np.ndarray, n: int, positive_only: bool = False) -> list[tuple[int, float]]:
    idx = np.flatnonzero(mask)
    if positive_only:
        idx = idx[scores[idx] > 0]
    order = idx[np.argsort(-scores[idx], kind="stable")][:n]
    return [(int(i), float(scores[i])) for i in order]


def rrf(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion : score(d) = somme sur les listes de 1 / (k + rang)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (i, _) in enumerate(ranking, start=1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


class Reranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[tuple[int, float]], index: Index) -> list[tuple[int, float]]:
        if not candidates:
            return []
        pairs = [(query, index.chunks[i]["embed_text"]) for i, _ in candidates]
        scores = self.model.predict(pairs)
        order = np.argsort(-np.asarray(scores))
        return [(candidates[j][0], float(scores[j])) for j in order]


class Retriever:
    def __init__(self, index: Index, embedder_getter: Callable | None = None, cfg: dict | None = None):
        self.index = index
        self._get_embedder = embedder_getter
        self.cfg = cfg or {}
        self._reranker: Reranker | None = None

    def _reranker_or_load(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker(self.cfg.get("reranker") or DEFAULT_RERANKER)
        return self._reranker

    def search(
        self,
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
        filters: dict | None = None,
        rerank: bool | None = None,
    ) -> list[Hit]:
        mode = mode or self.cfg.get("mode", "hybrid")
        top_k = top_k or self.cfg.get("top_k", 5)
        n_cand = max(self.cfg.get("candidates", 30), top_k)
        rerank = bool(self.cfg.get("reranker")) if rerank is None else rerank
        mask = self.index.mask(filters)

        rankings: list[list[tuple[int, float]]] = []
        if mode in ("bm25", "hybrid"):
            rankings.append(_top(self.index.bm25_scores(query), mask, n_cand, positive_only=True))
        if mode in ("dense", "hybrid"):
            if self._get_embedder is None:
                raise RuntimeError("Mode dense/hybrid demandé mais aucun modèle d'embeddings disponible.")
            qvec = self._get_embedder().encode_queries([query])[0]
            rankings.append(_top(self.index.dense_scores(qvec), mask, n_cand))
        if not rankings:
            raise ValueError(f"Mode inconnu : {mode!r} (bm25 | dense | hybrid)")

        candidates = rankings[0] if len(rankings) == 1 else rrf(rankings, self.cfg.get("rrf_k", 60))
        candidates = candidates[:n_cand]
        if rerank:
            candidates = self._reranker_or_load().rerank(query, candidates, self.index)
        return [Hit(self.index.chunks[i], s, r) for r, (i, s) in enumerate(candidates[:top_k], start=1)]
