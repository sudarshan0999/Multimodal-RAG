"""Extract plain text per page from PDF."""

from __future__ import annotations

import fitz


def extract_text_by_page(pdf_path: str) -> list[tuple[int, str]]:
    """Return list of (1-based page number, text)."""
    doc = fitz.open(pdf_path)
    pages: list[tuple[int, str]] = []
    try:
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text("text") or ""
            pages.append((i + 1, text))
    finally:
        doc.close()
    return pages
