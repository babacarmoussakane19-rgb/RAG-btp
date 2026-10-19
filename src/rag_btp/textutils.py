"""Utilitaires de normalisation de texte (français)."""
from __future__ import annotations

import re
import unicodedata

STOPWORDS_FR = frozenset(
    """
    a au aux avec ce ces dans de des du elle en et eux il je la le les leur lui ma mais me
    meme mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te
    tes toi ton tu un une vos votre vous c d j l m n s t y est sont ete etre avoir ont fait
    """.split()
)

_TOKEN = re.compile(r"[a-z0-9]+")


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Minuscules, sans accents, découpage alphanumérique (utilisé pour BM25)."""
    tokens = _TOKEN.findall(strip_accents(text.lower()))
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS_FR]
    return tokens


def normalize(text: str) -> str:
    """Forme normalisée pour comparer des textes (évaluation) sans tenir compte de la mise en forme."""
    return " ".join(tokenize(text, remove_stopwords=False))
