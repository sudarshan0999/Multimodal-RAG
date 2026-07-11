"""PDF table extraction using Camelot."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import camelot
import fitz

from providers.base import BaseLLMProvider


def get_table_summaries(
    pdf_path: str | Path,
    llm: BaseLLMProvider | None = None,
) -> list[dict[str, Any]]:
    """Extract tables from PDF and generate summaries using LLM.

    Returns list of dicts with keys: path, page, index, table_html, summary.
    """
    tables: list[dict[str, Any]] = []
    path = Path(pdf_path)

    try:
        camelot_tables = camelot.read_pdf(
            str(path),
            pages="all",
            flavor="lattice",
            suppress_messages=True,
        )
    except Exception as e:
        print(f"Camelot extraction error: {e}")
        try:
            camelot_tables = camelot.read_pdf(
                str(path),
                pages="all",
                flavor="stream",
                suppress_messages=True,
            )
        except Exception:
            camelot_tables = []

    for page_idx, page_tables in enumerate(camelot_tables):
        if not page_tables:
            continue
        page_num = page_idx + 1
        for table_idx, table in enumerate(page_tables.df.iterrows()):
            table_html = table_tables[table_idx].df.to_html()
            summary = ""
            if llm is not None:
                try:
                    prompt = f"""Describe this table in 1-2 sentences for retrieval.
Focus on: what data the table contains, column headers, and key insights.
Keep it brief."""
                    prompt += f"\n\nTable (HTML):\n{table_html[:1500]}"
                    summary = (
                        llm.generate_caption.__self__.generate_response(
                            [{"role": "user", "content": prompt}],
                            stream=False,
                        )
                        if hasattr(llm, "generate_caption")
                        else ""
                    )
                except Exception:
                    pass
            if not summary:
                cols = list(table.columns)
                summary = f"Table with {len(cols)} columns: {', '.join(cols[:5])}"

            tables.append(
                {
                    "page": page_num,
                    "index": table_idx + 1,
                    "table_html": table_html,
                    "summary": summary,
                }
            )

    return tables


def extract_tables_simple(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Simple table extraction without LLM summaries."""
    tables: list[dict[str, Any]] = []
    path = Path(pdf_path)

    try:
        camelot_tables = camelot.read_pdf(
            str(path),
            pages="all",
            flavor="lattice",
            suppress_messages=True,
        )
    except Exception:
        try:
            camelot_tables = camelot.read_pdf(
                str(path),
                pages="all",
                flavor="stream",
                suppress_messages=True,
            )
        except Exception:
            return []

    for page_idx, page_tables in enumerate(camelot_tables):
        if not page_tables:
            continue
        page_num = page_idx + 1
        for table_idx in range(len(page_tables)):
            table_df = page_tables[table_idx].df
            table_html = table_df.to_html()
            cols = list(table_df.columns)
            rows = len(table_df)
            summary = f"Table with {len(cols)} columns and {rows} rows"

            tables.append(
                {
                    "page": page_num,
                    "index": table_idx + 1,
                    "table_html": table_html,
                    "summary": summary,
                    "columns": cols,
                    "row_count": rows,
                }
            )

    return tables
