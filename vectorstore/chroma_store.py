"""ChromaDB persistence and CRUD for multimodal RAG documents."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from vectorstore.embedder import Embedder


class ChromaStore:
    def __init__(
        self,
        persist_directory: Path | str,
        collection_name: str = "multimodal_rag",
        embedder: Embedder | None = None,
    ) -> None:
        self._persist = Path(persist_directory)
        self._persist.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist))
        self._collection_name = collection_name
        self._embedder = embedder
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self) -> Collection:
        return self._collection

    def set_embedder(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def _ensure_embedder(self) -> Embedder:
        if self._embedder is None:
            raise RuntimeError("Embedder not set on ChromaStore")
        return self._embedder

    def delete_by_source_hash(self, source_hash: str) -> None:
        """Remove all chunks for a given source file hash."""
        col = self._collection
        data = col.get(where={"source_hash": source_hash}, include=[])
        ids = data.get("ids") or []
        if ids:
            col.delete(ids=ids)

    def has_source_hash(self, source_hash: str) -> bool:
        data = self._collection.get(
            where={"source_hash": source_hash},
            limit=1,
            include=[],
        )
        return bool(data.get("ids"))

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> list[str]:
        emb = self._ensure_embedder()
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        embeddings = emb.embed_documents(documents)
        # Chroma metadata values must be str, int, float, bool
        clean_meta: list[dict[str, Any]] = []
        for m in metadatas:
            cm: dict[str, Any] = {}
            for k, v in m.items():
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    cm[k] = v
                elif isinstance(v, Path):
                    cm[k] = str(v)
                else:
                    cm[k] = str(v)
            clean_meta.append(cm)
        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=clean_meta,
        )
        return ids

    def query_semantic(
        self,
        query_text: str,
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        emb = self._ensure_embedder()
        qemb = emb.embed_query(query_text)
        return self._collection.query(
            query_embeddings=[qemb],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def get_image_captions(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Return all stored image-caption rows: (id, caption_text, metadata)."""
        out: list[tuple[str, str, dict[str, Any]]] = []
        offset = 0
        batch = 500
        while True:
            batch_data = self._collection.get(
                where={"type": "image"},
                include=["documents", "metadatas"],
                limit=batch,
                offset=offset,
            )
            ids = batch_data.get("ids") or []
            if not ids:
                break
            docs = batch_data.get("documents") or []
            metas = batch_data.get("metadatas") or []
            for i, doc_id in enumerate(ids):
                doc = docs[i] if i < len(docs) else ""
                meta = metas[i] if i < len(metas) else {}
                out.append((doc_id, doc or "", dict(meta or {})))
            offset += len(ids)
        out.sort(
            key=lambda row: (
                str(row[2].get("source") or ""),
                int(row[2].get("page") or 0),
                row[0],
            )
        )
        return out

    def get_all_documents_for_bm25(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Return (id, document, metadata) for in-memory BM25 index."""
        out: list[tuple[str, str, dict[str, Any]]] = []
        offset = 0
        batch = 500
        while True:
            batch_data = self._collection.get(
                include=["documents", "metadatas"],
                limit=batch,
                offset=offset,
            )
            ids = batch_data.get("ids") or []
            if not ids:
                break
            docs = batch_data.get("documents") or []
            metas = batch_data.get("metadatas") or []
            for i, doc_id in enumerate(ids):
                doc = docs[i] if i < len(docs) else ""
                meta = metas[i] if i < len(metas) else {}
                out.append((doc_id, doc or "", dict(meta or {})))
            offset += len(ids)
        return out
