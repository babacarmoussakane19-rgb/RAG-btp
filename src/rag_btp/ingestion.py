"""Étape 1 : lecture des documents (PDF, Markdown, texte) et extraction de métadonnées.

Arborescence attendue : ``data/raw/<type_de_document>/<fichier>``. Le dossier de premier niveau
devient le champ ``doc_type`` (reglementation, technique, process_interne...).

Métadonnées facultatives :
- Markdown : front matter YAML en tête de fichier (title, date, version, statut...).
- PDF / autres : fichier compagnon ``<nom>.meta.yaml`` à côté du document.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SUPPORTED = {".md", ".txt", ".pdf"}

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.S)
# Titres « juridiques » (Article 3, Chapitre II...) ou numérotés (1.2 Contrôle du béton)
_PDF_HEADING = re.compile(
    r"^(?:(?i:(?:article|chapitre|titre|section|annexe)\s+[\w.\-]+.*)"
    r"|\d+(?:\.\d+)*[.)]?\s+[A-ZÀ-ÖØ-Þ].{2,80})$"
)


@dataclass
class Section:
    title: str
    text: str
    page: int | None = None
    kind: str = "text"  # "text" | "table"


@dataclass
class Document:
    doc_id: str
    source: str
    title: str
    metadata: dict
    sections: list[Section] = field(default_factory=list)


def _is_pdf_heading(line: str) -> bool:
    return len(line) <= 100 and not line.endswith((".", ";", ",")) and bool(_PDF_HEADING.match(line))


def _clean_meta(meta: dict) -> dict:
    """Convertit les dates YAML en chaînes pour rester sérialisable en JSON."""
    return {k: (str(v) if hasattr(v, "isoformat") else v) for k, v in meta.items()}


def parse_markdown(raw: str) -> tuple[dict, list[Section], str | None]:
    meta: dict = {}
    m = _FRONT_MATTER.match(raw)
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
        raw = m.group(2)

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    first_h1: str | None = None

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            title = " > ".join(t for _, t in stack) or "Introduction"
            sections.append(Section(title, text))
        buf.clear()

    for line in raw.splitlines():
        h = _MD_HEADING.match(line)
        if h:
            flush()
            level, title = len(h.group(1)), h.group(2)
            if level == 1 and first_h1 is None:
                first_h1 = title
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return _clean_meta(meta), sections, first_h1


def _rows_to_markdown(rows: list[list]) -> str:
    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in rows if r]
    if len(rows) < 2:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def parse_pdf(path: Path) -> list[Section]:
    """Texte par page avec détection heuristique des titres + tableaux en Markdown.

    Limites connues : pas d'OCR (PDF scannés), mise en page multi-colonnes approximative,
    les tableaux sont aussi présents dans le texte brut de la page (doublon assumé).
    """
    import fitz  # PyMuPDF

    sections: list[Section] = []
    current = "Introduction"
    buf: list[str] = []
    buf_page = 1

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(current, text, buf_page))
        buf.clear()

    with fitz.open(path) as pdf:
        for pno, page in enumerate(pdf, start=1):
            for line in page.get_text("text").splitlines():
                line = line.strip()
                if not line:
                    continue
                if _is_pdf_heading(line):
                    flush()
                    current, buf_page = line, pno
                else:
                    if not buf:
                        buf_page = pno
                    buf.append(line)
            flush()
            try:
                for table in page.find_tables().tables:
                    md = _rows_to_markdown(table.extract())
                    if md:
                        sections.append(Section(f"{current} (tableau)", md, pno, "table"))
            except Exception:  # find_tables absent ou échec sur une page : on continue
                pass
    return sections


def load_documents(raw_dir: str | Path) -> list[Document]:
    raw = Path(raw_dir)
    docs: list[Document] = []
    for path in sorted(raw.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        if path.name.lower().startswith("readme"):
            continue
        rel = path.relative_to(raw)
        meta = {
            "doc_type": rel.parts[0] if len(rel.parts) > 1 else "autre",
            "date": None,
            "version": None,
            "statut": "inconnu",
        }
        first_h1 = None
        if path.suffix.lower() == ".pdf":
            sections = parse_pdf(path)
        else:
            file_meta, sections, first_h1 = parse_markdown(path.read_text(encoding="utf-8"))
            meta.update(file_meta)
        sidecar = path.with_suffix(".meta.yaml")
        if sidecar.exists():
            meta.update(_clean_meta(yaml.safe_load(sidecar.read_text(encoding="utf-8")) or {}))
        title = meta.pop("title", None) or first_h1 or path.stem.replace("_", " ")
        docs.append(Document(rel.with_suffix("").as_posix(), rel.as_posix(), title, meta, sections))
    if not docs:
        warnings.warn(f"Aucun document trouvé dans {raw} (formats : {sorted(SUPPORTED)})")
    return docs
