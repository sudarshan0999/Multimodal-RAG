"""Semantic chunking via embedding similarity breakpoints."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from chunking.base import BaseChunker

if TYPE_CHECKING:
    from vectorstore.embedder import BaseEmbedder


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


class SemanticChunker(BaseChunker):
    """
    Group sentences into chunks where cosine similarity between adjacent
    windows stays above a threshold; uses the same embedder as retrieval.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        embedder: BaseEmbedder | None = None,
        similarity_drop_threshold: float = 0.3,
        min_sentences_per_chunk: int = 1,
    ) -> None:
        super().__init__(chunk_size, overlap)
        self._embedder = embedder
        self.similarity_drop_threshold = similarity_drop_threshold
        self.min_sentences_per_chunk = min_sentences_per_chunk

    def set_embedder(self, embedder: BaseEmbedder) -> None:
        self._embedder = embedder

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
        sents = _sentences(text)
        if len(sents) <= self.min_sentences_per_chunk:
            return [text.strip()] if text.strip() else []

        if self._embedder is None:
            from chunking.fixed import FixedChunker

            return FixedChunker(self.chunk_size, self.overlap).chunk(text)

        # Embed each sentence
        emb_list = self._embedder.embed_documents(sents)
        arr = np.array(emb_list, dtype=np.float64)

        # Breakpoints where similarity to next sentence drops
        breakpoints = [0]
        for i in range(len(arr) - 1):
            sim = self._cosine(arr[i], arr[i + 1])
            if sim < (1.0 - self.similarity_drop_threshold):
                breakpoints.append(i + 1)
        breakpoints.append(len(sents))

        raw_chunks: list[str] = []
        for bi in range(len(breakpoints) - 1):
            start_i = breakpoints[bi]
            end_i = breakpoints[bi + 1]
            chunk_text = " ".join(sents[start_i:end_i])
            if chunk_text.strip():
                raw_chunks.append(chunk_text)

        # Enforce max character size with fixed chunker
        from chunking.fixed import FixedChunker

        fixed = FixedChunker(self.chunk_size, self.overlap)
        out: list[str] = []
        for rc in raw_chunks:
            out.extend(fixed.chunk(rc))
        return out
