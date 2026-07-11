from abc import ABC, abstractmethod


class BaseChunker(ABC):
    """Split plain text into chunks for embedding."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200) -> None:
        self.chunk_size = max(1, chunk_size)
        self.overlap = max(0, min(overlap, self.chunk_size - 1))

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        ...
