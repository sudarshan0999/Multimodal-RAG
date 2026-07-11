"""RAG prompt construction and answer generation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from providers.base import BaseLLMProvider
from retrieval.hybrid import HybridRetriever, RetrievedDocument


SYSTEM_PROMPT = """You are a precise technical assistant. Answer questions using ONLY the provided context.

IMPORTANT GUIDELINES:
1. Always cite your sources using the format [Page X, Type Y] after any factual statement
2. If the exact information is not in the context, say "I don't have that information" - do NOT guess
3. For tables, cite the column names and specific values (e.g. "According to the table on page 8, column 'Revenue' shows...")
4. For images/diagrams, cite them as "See the diagram on page X"
5. Be concise and factual - list key points with bullet points
6. If multiple pages have relevant info, cite all of them"""


def build_user_prompt(
    query: str,
    retrieved: list[RetrievedDocument],
    related_content: list[RetrievedDocument] | None = None,
) -> str:
    text_blocks: list[str] = []
    table_blocks: list[str] = []
    diagram_blocks: list[str] = []
    vector_blocks: list[str] = []

    for i, r in enumerate(retrieved, start=1):
        meta = r.metadata
        typ = meta.get("type", "text")
        page = meta.get("page", "?")
        if typ == "image":
            ip = meta.get("image_path", "")
            diagram_blocks.append(
                f"[Image {i}] (page {page}, file: {ip})\nCaption: {r.document}\n"
            )
        elif typ == "table":
            html = meta.get("table_html", "")
            cols = meta.get("columns", [])
            table_blocks.append(
                f"[Table {i}] (page {page})\nColumns: {cols}\nSummary: {r.document}\n"
            )
        elif typ == "vector_graphic":
            ip = meta.get("image_path", "")
            vector_blocks.append(
                f"[Vector {i}] (page {page}, file: {ip})\nCaption: {r.document}\n"
            )
        else:
            text_blocks.append(f"[Text {i}] (page {page})\n{r.document}\n")

    if related_content:
        for r in related_content:
            meta = r.metadata
            typ = meta.get("type", "text")
            page = meta.get("page", "?")
            if typ == "table" and r not in retrieved:
                html = meta.get("table_html", "")
                cols = meta.get("columns", [])
                table_blocks.append(
                    f"[Table from page {page}]\nColumns: {cols}\nSummary: {r.document}\n"
                )
            elif typ == "image" and r not in retrieved:
                ip = meta.get("image_path", "")
                diagram_blocks.append(
                    f"[Image from page {page}, file: {ip}]\nCaption: {r.document}\n"
                )

    parts = [
        "## User question\n",
        query.strip(),
        "\n\n## Text context\n",
        "\n".join(text_blocks) if text_blocks else "(none)",
        "\n\n## Table context (from same pages)\n",
        "\n".join(table_blocks) if table_blocks else "(none)",
        "\n\n## Diagram / image context\n",
        "\n".join(diagram_blocks) if diagram_blocks else "(none)",
    ]
    return "".join(parts)


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, llm: BaseLLMProvider) -> None:
        self._retriever = retriever
        self._llm = llm

    def retrieve_only(self, query: str, top_k: int) -> list[RetrievedDocument]:
        return self._retriever.retrieve(query, top_k=top_k)

    def answer(
        self,
        query: str,
        *,
        top_k: int = 8,
        stream: bool = False,
        include_related: bool = True,
    ) -> (
        tuple[str, list[RetrievedDocument]]
        | tuple[Iterator[str], list[RetrievedDocument]]
    ):
        retrieved = self.retrieve_only(query, top_k)

        related_content = []
        if include_related:
            related_content = self._retriever.get_related_by_page(
                query, retrieved, include_types=["table", "image", "vector_graphic"]
            )

        user_content = build_user_prompt(query, retrieved, related_content)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        if stream:
            return self._llm.generate_response(messages, stream=True), retrieved
        text = self._llm.generate_response(messages, stream=False)
        assert isinstance(text, str)
        return text, retrieved

    @staticmethod
    def image_paths_from_retrieval(retrieved: list[RetrievedDocument]) -> list[str]:
        paths: list[str] = []
        for r in retrieved:
            if r.metadata.get("type") == "image":
                p = r.metadata.get("image_path")
                if p and Path(p).is_file():
                    paths.append(str(p))
        return paths
