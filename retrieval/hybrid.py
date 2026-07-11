"""Hybrid retrieval: dense vectors + BM25 with RRF-style fusion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from retrieval.reranker import BaseReranker, get_reranker, RerankerKind
from vectorstore.chroma_store import ChromaStore


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class RetrievedDocument:
    id: str
    document: str
    metadata: dict[str, Any]
    semantic_score: float | None
    bm25_score: float | None
    fused_score: float


class HybridRetriever:
    def __init__(
        self,
        store: ChromaStore,
        *,
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60,
        image_top_k: int = 4,
        image_semantic_pool: int = 32,
        reranker: BaseReranker | None = None,
    ) -> None:
        self._store = store
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self._image_top_k = image_top_k
        self._image_semantic_pool = image_semantic_pool
        self._reranker = reranker

    def set_reranker(self, reranker: BaseReranker | None) -> None:
        self._reranker = reranker

    def get_all_by_page(
        self,
        reference_docs: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """Get ALL content (text, table, image) from pages in reference_docs.

        If top_k returns page 22, this fetches ALL content from page 22:
        - all text chunks
        - all tables
        - all images
        This ensures full page context is available to LLM.
        """
        if not reference_docs:
            return []

        pages = set()
        for d in reference_docs:
            pg = d.metadata.get("page")
            if pg:
                pages.add(int(pg))

        if not pages:
            return []

        store = self._store
        all_content_types = ["text", "table", "image"]

        related: list[RetrievedDocument] = []

        for p in pages:
            for typ in all_content_types:
                try:
                    results = store.query_semantic(
                        f"page {p}",
                        n_results=50,
                        where={"type": typ, "page": p},
                    )
                except Exception:
                    continue

                ids = (results.get("ids") or [[]])[0]
                docs = (results.get("documents") or [[]])[0]
                metas = (results.get("metadatas") or [[]])[0]
                dists = (results.get("distances") or [[]])[0]

                for i, doc_id in enumerate(ids):
                    if i >= len(docs):
                        break
                    dist = dists[i] if i < len(dists) else 1.0
                    sim = 1.0 / (1.0 + float(dist))
                    meta = dict(metas[i] if i < len(metas) else {})
                    related.append(
                        RetrievedDocument(
                            id=doc_id,
                            document=docs[i] or "",
                            metadata=meta,
                            semantic_score=sim,
                            bm25_score=None,
                            fused_score=sim,
                        )
                    )

        seen = set()
        unique: list[RetrievedDocument] = []
        for r in related:
            if r.id not in seen:
                seen.add(r.id)
                unique.append(r)

        return unique

    def get_related_by_page(
        self,
        query: str,
        reference_docs: list[RetrievedDocument],
        *,
        include_types: list[str] | None = None,
    ) -> list[RetrievedDocument]:
        """Get all content from pages matching the reference docs.

        If reference_docs include pages from a query, this fetches all tables,
        images, vector_graphics from those same pages.
        """
        if not reference_docs:
            return []

        pages = set()
        for d in reference_docs:
            pg = d.metadata.get("page")
            if pg:
                pages.add(int(pg))

        if not pages:
            return []

        store = self._store
        if include_types is None:
            include_types = ["table", "image", "vector_graphic"]

        related: list[RetrievedDocument] = []

        for typ in include_types:
            for p in pages:
                try:
                    results = store.query_semantic(
                        f"page {p} {typ}",
                        n_results=10,
                        where={"type": typ, "page": p},
                    )
                except Exception:
                    continue

                ids = (results.get("ids") or [[]])[0]
                docs = (results.get("documents") or [[]])[0]
                metas = (results.get("metadatas") or [[]])[0]
                dists = (results.get("distances") or [[]])[0]

                for i, doc_id in enumerate(ids):
                    if i >= len(docs):
                        break
                    dist = dists[i] if i < len(dists) else 1.0
                    sim = 1.0 / (1.0 + float(dist))
                    meta = dict(metas[i] if i < len(metas) else {})
                    related.append(
                        RetrievedDocument(
                            id=doc_id,
                            document=docs[i] or "",
                            metadata=meta,
                            semantic_score=sim,
                            bm25_score=None,
                            fused_score=sim,
                        )
                    )

        seen = set()
        unique: list[RetrievedDocument] = []
        for r in related:
            if r.id not in seen:
                seen.add(r.id)
                unique.append(r)

        return unique

    def _apply_reranking(
        self,
        query: str,
        docs: list[RetrievedDocument],
        top_k: int,
    ) -> list[RetrievedDocument]:
        if not self._reranker or not docs:
            return docs[:top_k]
        doc_texts = [d.document for d in docs]
        reranked = self._reranker.rerank(query, doc_texts, top_k=top_k)
        reranked_docs = []
        for orig_idx, score in reranked:
            d = docs[orig_idx]
            reranked_docs.append(
                RetrievedDocument(
                    id=d.id,
                    document=d.document,
                    metadata=d.metadata,
                    semantic_score=d.semantic_score,
                    bm25_score=d.bm25_score,
                    fused_score=float(score),
                )
            )
        return reranked_docs

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        fetch_k: int = 40,
    ) -> list[RetrievedDocument]:
        """Retrieve top_k (hybrid text+all) plus extra image-caption rows from image-only semantic search.

        Plain hybrid retrieval often ranks text chunks above figure captions, so queries rarely
        surface diagrams. When ``image_top_k`` > 0 (set on the retriever), we append the best
        matching ``type=image`` documents by embedding similarity (excluding ids already in the
        hybrid slice).
        """
        image_top_k = self._image_top_k
        image_semantic_pool = self._image_semantic_pool
        store = self._store
        # Semantic
        sem_raw = store.query_semantic(query, n_results=min(fetch_k, max(top_k, 20)))
        sem_ids = (sem_raw.get("ids") or [[]])[0]
        sem_docs = (sem_raw.get("documents") or [[]])[0]
        sem_metas = (sem_raw.get("metadatas") or [[]])[0]
        sem_dists = (sem_raw.get("distances") or [[]])[0]

        # distance lower is better (cosine distance in chroma) -> similarity
        sem_rank: dict[str, tuple[int, float]] = {}
        for rank, doc_id in enumerate(sem_ids):
            dist = sem_dists[rank] if rank < len(sem_dists) else 1.0
            sim = 1.0 / (1.0 + float(dist))
            sem_rank[doc_id] = (rank, sim)

        # BM25 index over full collection (in-memory rebuild each query — OK for small/medium corpora)
        all_rows = store.get_all_documents_for_bm25()
        corpus_tokens = [_tokenize(doc) for _, doc, _ in all_rows]
        id_list = [doc_id for doc_id, _, _ in all_rows]
        if corpus_tokens:
            bm25 = BM25Okapi(corpus_tokens)
            q_tokens = _tokenize(query)
            scores = bm25.get_scores(q_tokens) if q_tokens else [0.0] * len(id_list)
        else:
            scores = []

        bm25_rank: dict[str, tuple[int, float]] = {}
        if id_list and scores is not None and len(scores) == len(id_list):
            ranked = sorted(
                zip(id_list, list(scores)),
                key=lambda x: x[1],
                reverse=True,
            )
            for r, (doc_id, sc) in enumerate(ranked[:fetch_k]):
                bm25_rank[doc_id] = (r, float(sc))

        # Normalize BM25 scores 0..1 within top pool
        if bm25_rank:
            max_s = max(s for _, s in bm25_rank.values()) or 1.0
            bm25_rank = {k: (r, v / max_s) for k, (r, v) in bm25_rank.items()}

        # Candidate union
        candidates = set(sem_rank) | set(bm25_rank)
        # RRF
        k = self.rrf_k
        fused: dict[str, float] = {}
        for doc_id in candidates:
            rrf = 0.0
            if doc_id in sem_rank:
                rrf += self.semantic_weight / (k + sem_rank[doc_id][0] + 1)
            if doc_id in bm25_rank:
                rrf += self.bm25_weight / (k + bm25_rank[doc_id][0] + 1)
            fused[doc_id] = rrf

        sorted_ids = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)[:top_k]

        # Build doc map
        by_id: dict[str, tuple[str, dict[str, Any]]] = {
            doc_id: (doc, meta) for doc_id, doc, meta in all_rows
        }
        # Fill from semantic result if missing in full scan (should not happen)
        for i, doc_id in enumerate(sem_ids):
            if doc_id not in by_id and i < len(sem_docs):
                meta = sem_metas[i] if i < len(sem_metas) else {}
                by_id[doc_id] = (sem_docs[i] or "", dict(meta or {}))

        out: list[RetrievedDocument] = []
        for doc_id in sorted_ids:
            doc, meta = by_id.get(doc_id, ("", {}))
            sr = sem_rank.get(doc_id)
            br = bm25_rank.get(doc_id)
            out.append(
                RetrievedDocument(
                    id=doc_id,
                    document=doc,
                    metadata=meta,
                    semantic_score=sr[1] if sr else None,
                    bm25_score=br[1] if br else None,
                    fused_score=fused[doc_id],
                )
            )

        if image_top_k <= 0:
            return out

        seen = {r.id for r in out}
        try:
            img_raw = store.query_semantic(
                query,
                n_results=min(image_semantic_pool, max(image_top_k * 4, 8)),
                where={"type": "image"},
            )
        except Exception:
            return out

        img_ids = (img_raw.get("ids") or [[]])[0]
        img_docs = (img_raw.get("documents") or [[]])[0]
        img_metas = (img_raw.get("metadatas") or [[]])[0]
        img_dists = (img_raw.get("distances") or [[]])[0]

        added = 0
        for i, doc_id in enumerate(img_ids):
            if doc_id in seen:
                continue
            dist = img_dists[i] if i < len(img_dists) else 1.0
            sim = 1.0 / (1.0 + float(dist))
            doc = img_docs[i] if i < len(img_docs) else ""
            meta = dict(img_metas[i] if i < len(img_metas) else {})
            out.append(
                RetrievedDocument(
                    id=doc_id,
                    document=doc or "",
                    metadata=meta,
                    semantic_score=sim,
                    bm25_score=None,
                    fused_score=sim * 0.99,
                )
            )
            seen.add(doc_id)
            added += 1
            if added >= image_top_k:
                break

        if self._reranker:
            out = self._apply_reranking(query, out, top_k)

        return out
