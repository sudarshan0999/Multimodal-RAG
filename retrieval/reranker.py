"""Cross-encoder reranking for improved retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from sentence_transformers import CrossEncoder

RerankerKind = Literal["none", "ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-v2-m3"]


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Rerank documents given a query.

        Returns list of (index, score) tuples, sorted by score descending.
        """
        ...


class NoOpReranker(BaseReranker):
    """Pass-through reranker that returns original order with mock scores."""

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        n = min(len(documents), top_k) if top_k else len(documents)
        return [(i, 1.0 / (i + 1)) for i in range(n)]


class CrossEncoderReranker(BaseReranker):
    """Sentence Transformers cross-encoder reranker."""

    def __init__(self, model_name: str) -> None:
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        indexed = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )
        if top_k:
            indexed = indexed[:top_k]
        return indexed


def get_reranker(kind: RerankerKind) -> BaseReranker:
    if kind == "none":
        return NoOpReranker()
    if kind == "ms-marco-MiniLM-L-6-v2":
        return CrossEncoderReranker("ms-marco-MiniLM-L-6-v2")
    if kind == "BAAI/bge-reranker-v2-m3":
        return CrossEncoderReranker("BAAI/bge-reranker-v2-m3")
    raise ValueError(f"Unknown reranker: {kind}")
