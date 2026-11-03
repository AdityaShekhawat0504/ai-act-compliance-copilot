"""Load product documents and verify quotes against them.

Supported inputs: .md, .markdown, .txt and .pdf. PDF text is extracted with
pypdf (imported lazily so the rest of the package works without it).
"""

from __future__ import annotations

import io
import re
from pathlib import Path

from pydantic import BaseModel

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
PDF_SUFFIXES = {".pdf"}
MIN_QUOTE_CHARS = 12
"""Quotes shorter than this ('yes', 'AI') prove nothing and are rejected."""


class SourceDocument(BaseModel):
    name: str
    text: str


def _pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Reading PDF input needs pypdf (installed by `uv sync`).") from exc
    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def document_from_bytes(name: str, data: bytes) -> SourceDocument:
    suffix = Path(name).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return SourceDocument(name=name, text=data.decode("utf-8", errors="replace"))
    if suffix in PDF_SUFFIXES:
        return SourceDocument(name=name, text=_pdf_text(data))
    raise ValueError(f"Unsupported document type {suffix!r} for {name}; use .md, .txt or .pdf")


def load_document(path: Path) -> SourceDocument:
    path = Path(path)
    return document_from_bytes(path.name, path.read_bytes())


_WS = re.compile(r"\s+")
_QUOTE_CHARS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def _normalise(text: str) -> str:
    """Collapse whitespace and unify curly quotes. Nothing else is changed:
    case, words and punctuation must match exactly."""
    return _WS.sub(" ", text.translate(_QUOTE_CHARS)).strip()


def verify_quote(quote: str | None, text: str) -> bool:
    """True if *quote* appears verbatim in *text*.

    Only whitespace runs (line breaks in PDFs, wrapped Markdown) and curly
    versus straight quote marks are normalised.
    """
    if not quote:
        return False
    q = _normalise(quote)
    if len(q) < MIN_QUOTE_CHARS:
        return False
    return q in _normalise(text)
