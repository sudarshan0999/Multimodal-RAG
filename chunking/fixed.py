from chunking.base import BaseChunker


class FixedChunker(BaseChunker):
    """Fixed-size character windows with overlap."""

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
        size, ov = self.chunk_size, self.overlap
        step = max(1, size - ov)
        chunks: list[str] = []
        for start in range(0, len(text), step):
            piece = text[start : start + size]
            if piece.strip():
                chunks.append(piece)
            if start + size >= len(text):
                break
        return chunks
