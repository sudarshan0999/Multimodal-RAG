import tiktoken

from chunking.base import BaseChunker


class TokenChunker(BaseChunker):
    """Token-based windows using tiktoken (cl100k_base default)."""

    def __init__(
        self,
        chunk_size: int = 256,
        overlap: int = 32,
        encoding_name: str = "cl100k_base",
    ) -> None:
        super().__init__(chunk_size, overlap)
        self._enc = tiktoken.get_encoding(encoding_name)

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
        tokens = self._enc.encode(text)
        size, ov = self.chunk_size, self.overlap
        step = max(1, size - ov)
        chunks: list[str] = []
        for start in range(0, len(tokens), step):
            slice_tok = tokens[start : start + size]
            if not slice_tok:
                break
            chunks.append(self._enc.decode(slice_tok))
            if start + size >= len(tokens):
                break
        return [c for c in chunks if c.strip()]
