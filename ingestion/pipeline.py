"""Orchestrate PDF text + image + table extraction, captioning, and vector indexing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from chunking.base import BaseChunker
from chunking.factory import get_chunker
from config.settings import Settings
from ingestion.pdf_images import fetch_images_from_pdf
from ingestion.pdf_tables import extract_tables_simple
from ingestion.pdf_text import extract_text_by_page
from providers.base import BaseLLMProvider
from utils.cache import source_file_hash
from vectorstore.chroma_store import ChromaStore
from vectorstore.embedder import Embedder, build_embedder


StatusCallback = Callable[[str], None]


@dataclass
class IngestionResult:
    source: str
    source_hash: str
    text_chunks: int
    image_records: int
    table_records: int
    skipped: bool
    image_paths: list[str] = field(default_factory=list)


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        store: ChromaStore,
        llm: BaseLLMProvider,
        embedder: Embedder | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._llm = llm
        self._embedder = embedder

    def _embedder_instance(self, embed_backend: str | None) -> Embedder:
        if embed_backend is None and self._embedder is not None:
            return self._embedder
        inner = build_embedder(self._settings, override_backend=embed_backend)
        return Embedder(inner)

    def ingest_pdf(
        self,
        pdf_path: str | Path,
        *,
        source_name: str | None = None,
        chunker: BaseChunker | None = None,
        chunk_strategy: str = "fixed",
        chunk_size: int = 1000,
        overlap: int = 200,
        force_reindex: bool = False,
        embed_backend: str | None = None,
        semantic_embedder=None,
        on_status: StatusCallback | None = None,
    ) -> IngestionResult:
        path = Path(pdf_path)
        name = source_name or path.name
        data = path.read_bytes()
        shash = source_file_hash(data)

        def status(msg: str) -> None:
            if on_status:
                on_status(msg)

        emb = self._embedder_instance(embed_backend)
        self._store.set_embedder(emb)

        if not force_reindex and self._store.has_source_hash(shash):
            status("Skipping ingest: already indexed (use re-index to force).")
            return IngestionResult(
                source=name,
                source_hash=shash,
                text_chunks=0,
                image_records=0,
                table_records=0,
                skipped=True,
            )

        if force_reindex:
            status("Removing old vectors for this file hash…")
            self._store.delete_by_source_hash(shash)

        if chunker is None:
            chunker = get_chunker(
                chunk_strategy,  # type: ignore[arg-type]
                chunk_size,
                overlap,
                semantic_embedder=semantic_embedder,
            )

        status("Extracting text…")
        page_texts = extract_text_by_page(str(path))

        text_docs: list[str] = []
        text_metas: list[dict[str, Any]] = []
        chunk_idx = 0
        for page_num, ptext in page_texts:
            if not (ptext or "").strip():
                continue
            for chunk in chunker.chunk(ptext):
                if not chunk.strip():
                    continue
                text_docs.append(chunk)
                text_metas.append(
                    {
                        "source": name,
                        "source_hash": shash,
                        "page": page_num,
                        "type": "text",
                        "chunk_index": chunk_idx,
                        "image_path": "",
                        "section": "",
                        "subsection": "",
                        "keywords": [],
                    }
                )
                chunk_idx += 1

        if text_docs:
            status(f"Indexing {len(text_docs)} text chunks…")
            self._store.add_documents(text_docs, text_metas)

        img_dir = self._settings.captions_dir / shash[:16]
        img_dir.mkdir(parents=True, exist_ok=True)
        status("Extracting diagrams and images…")
        saved = fetch_images_from_pdf(str(path), str(img_dir))
        image_paths = [s["path"] for s in saved]

        cap_docs: list[str] = []
        cap_metas: list[dict[str, Any]] = []
        for meta in saved:
            ip = meta["path"]
            page_num = meta["page"]
            status(f"Captioning image (page {page_num})…")
            try:
                caption = self._llm.generate_caption(ip)
            except Exception as e:
                caption = f"[Caption failed: {e}]"
            cap_docs.append(caption)
            cap_metas.append(
                {
                    "source": name,
                    "source_hash": shash,
                    "page": page_num,
                    "type": "image",
                    "image_path": str(Path(ip).resolve()),
                    "chunk_index": -1,
                    "section": "",
                    "subsection": "",
                    "keywords": [],
                }
            )

        if cap_docs:
            status(f"Indexing {len(cap_docs)} image captions…")
            self._store.add_documents(cap_docs, cap_metas)

        status("Extracting tables…")
        tables = extract_tables_simple(str(path))

        tab_docs: list[str] = []
        tab_metas: list[dict[str, Any]] = []
        for tab in tables:
            tab_docs.append(tab["summary"])
            tab_metas.append(
                {
                    "source": name,
                    "source_hash": shash,
                    "page": tab["page"],
                    "type": "table",
                    "table_html": tab.get("table_html", ""),
                    "image_path": "",
                    "chunk_index": -1,
                    "section": "",
                    "subsection": "",
                    "keywords": [],
                    "columns": tab.get("columns", []),
                    "row_count": tab.get("row_count", 0),
                }
            )

        if tab_docs:
            status(f"Indexing {len(tab_docs)} table summaries…")
            self._store.add_documents(tab_docs, tab_metas)

        return IngestionResult(
            source=name,
            source_hash=shash,
            text_chunks=len(text_docs),
            image_records=len(cap_docs),
            table_records=len(tab_docs),
            skipped=False,
            image_paths=image_paths,
        )
