from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from chunking.base import BaseChunker
from chunking.fixed import FixedChunker
from chunking.recursive import RecursiveChunker
from chunking.semantic import SemanticChunker
from chunking.token import TokenChunker

if TYPE_CHECKING:
    from vectorstore.embedder import BaseEmbedder

ChunkStrategy = Literal["fixed", "recursive", "token", "semantic"]


def get_chunker(
    strategy: ChunkStrategy,
    chunk_size: int = 1000,
    overlap: int = 200,
    semantic_embedder: BaseEmbedder | None = None,
) -> BaseChunker:
    if strategy == "fixed":
        return FixedChunker(chunk_size, overlap)
    if strategy == "recursive":
        return RecursiveChunker(chunk_size, overlap)
    if strategy == "token":
        return TokenChunker(chunk_size, overlap)
    if strategy == "semantic":
        sc = SemanticChunker(chunk_size, overlap)
        if semantic_embedder is not None:
            sc.set_embedder(semantic_embedder)
        return sc
    raise ValueError(f"Unknown chunk strategy: {strategy}")
