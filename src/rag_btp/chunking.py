"""Étape 2 : découpage en chunks par section, avec chevauchement et métadonnées héritées.

Principes :
- on ne découpe jamais à travers un titre : chaque section est découpée séparément ;
- l'unité atomique est la ligne (un tableau Markdown n'est jamais coupé au milieu d'une ligne) ;
- chaque chunk porte son contexte (titre du document + section) dans ``embed_text``.
"""
from __future__ import annotations

from .ingestion import Document

META_FIELDS = ("doc_type", "date", "version", "statut")


def _split_units(text: str, max_words: int) -> list[str]:
    units: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        words = line.split()
        if len(words) <= max_words:
            units.append(line)
        else:  # ligne géante (PDF sans retours à la ligne) : découpage forcé par mots
            units += [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]
    return units


def pack_units(units: list[str], max_words: int, overlap_words: int) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    for unit in units:
        uw = len(unit.split())
        if cur and n + uw > max_words:
            chunks.append("\n".join(cur))
            keep: list[str] = []
            k = 0
            for prev in reversed(cur):
                pw = len(prev.split())
                if k + pw > overlap_words:
                    break
                keep.insert(0, prev)
                k += pw
            cur, n = keep, k
        cur.append(unit)
        n += uw
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def chunk_document(doc: Document, max_words: int = 220, overlap_words: int = 40) -> list[dict]:
    chunks: list[dict] = []
    for section in doc.sections:
        units = _split_units(section.text, max_words)
        for text in pack_units(units, max_words, overlap_words):
            chunk = {
                "chunk_id": f"{doc.doc_id}::{len(chunks)}",
                "doc_id": doc.doc_id,
                "source": doc.source,
                "title": doc.title,
                "section": section.title,
                "page": section.page,
                "kind": section.kind,
                "text": text,
                "embed_text": f"{doc.title} | {section.title}\n{text}",
            }
            for f in META_FIELDS:
                chunk[f] = doc.metadata.get(f)
            chunks.append(chunk)
    return chunks


def chunk_documents(docs: list[Document], max_words: int = 220, overlap_words: int = 40) -> list[dict]:
    out: list[dict] = []
    for d in docs:
        out += chunk_document(d, max_words, overlap_words)
    return out
