"""Orchestration : construction de l'index, retrieval, génération."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

from .chunking import chunk_documents
from .config import load_config
from .embeddings import Embedder
from .generation import generate_answer, source_label
from .index import Index
from .ingestion import load_documents
from .retrieval import Hit, Retriever


@dataclass
class Answer:
    question: str
    answer: str | None
    hits: list[Hit]

    def sources_markdown(self) -> str:
        return "\n".join(f"[{h.rank}] {source_label(h.chunk)}" for h in self.hits)


class RAGPipeline:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.index: Index | None = None
        self._embedder: Embedder | None = None
        self._retriever: Retriever | None = None

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder.from_config(self.cfg)
        return self._embedder

    def build_index(self, embed: bool = True) -> Index:
        docs = load_documents(self.cfg["paths"]["raw_dir"])
        c = self.cfg["chunking"]
        chunks = chunk_documents(docs, c["max_words"], c["overlap_words"])
        embeddings, model = None, None
        if embed and chunks:
            embeddings = self.embedder.encode_passages([ch["embed_text"] for ch in chunks])
            model = self.cfg["embedding"]["model"]
        self.index = Index(chunks, embeddings, model)
        self.index.save(self.cfg["paths"]["index_dir"])
        self._retriever = None
        print(f"{len(docs)} documents -> {len(chunks)} chunks (embeddings : {'oui' if embeddings is not None else 'non'})")
        return self.index

    def load_index(self) -> Index:
        self.index = Index.load(self.cfg["paths"]["index_dir"])
        wanted = self.cfg["embedding"]["model"]
        if self.index.embedding_model and self.index.embedding_model != wanted:
            warnings.warn(
                f"Index construit avec {self.index.embedding_model} mais config = {wanted} : reconstruire l'index."
            )
        self._retriever = None
        return self.index

    @property
    def retriever(self) -> Retriever:
        if self.index is None:
            self.load_index()
        if self._retriever is None:
            self._retriever = Retriever(self.index, lambda: self.embedder, self.cfg["retrieval"])
        return self._retriever

    def retrieve(self, question: str, **kwargs) -> list[Hit]:
        return self.retriever.search(question, **kwargs)

    def ask(self, question: str, use_llm: bool = True, **kwargs) -> Answer:
        hits = self.retrieve(question, **kwargs)
        answer = generate_answer(question, hits, self.cfg["generation"]) if use_llm else None
        return Answer(question, answer, hits)
