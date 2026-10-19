"""Chargement de la configuration (défauts + YAML + variables d'environnement)."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "paths": {"raw_dir": "data/raw", "index_dir": "data/index"},
    "chunking": {"max_words": 220, "overlap_words": 40},
    "embedding": {
        "model": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "batch_size": 32,
    },
    "retrieval": {
        "mode": "hybrid",
        "top_k": 5,
        "candidates": 30,
        "rrf_k": 60,
        "reranker": None,
    },
    "generation": {"model": "claude-sonnet-5-5", "max_tokens": 800, "temperature": None},
}


def _merge(base: dict, override: dict | None) -> dict:
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | os.PathLike | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    candidate = Path(path) if path else Path("configs/config.yaml")
    if candidate.exists():
        _merge(cfg, yaml.safe_load(candidate.read_text(encoding="utf-8")))
    elif path:
        raise FileNotFoundError(f"Config introuvable : {path}")
    if os.getenv("RAG_RAW_DIR"):
        cfg["paths"]["raw_dir"] = os.environ["RAG_RAW_DIR"]
    if os.getenv("RAG_INDEX_DIR"):
        cfg["paths"]["index_dir"] = os.environ["RAG_INDEX_DIR"]
    return cfg
