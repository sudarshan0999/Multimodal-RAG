from chunking.base import BaseChunker
from chunking.fixed import FixedChunker

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker(BaseChunker):
    """Recursively split on separators (LangChain-style), then merge to chunk_size."""

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        super().__init__(chunk_size, overlap)
        self.separators = separators or list(_DEFAULT_SEPARATORS)

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        final_chunks: list[str] = []
        sep = separators[0]
        if len(separators) == 1:
            return self._fixed_split(text)
        splits = text.split(sep) if sep else list(text)
        good: list[str] = []
        for s in splits:
            if self._len(s) <= self.chunk_size:
                good.append(s)
            else:
                good.extend(self._split_text(s, separators[1:]))
        merged: list[str] = []
        cur = ""
        for g in good:
            if not g:
                continue
            if not cur:
                cur = g
            elif self._len(cur) + self._len(sep) + self._len(g) <= self.chunk_size:
                cur = cur + sep + g
            else:
                merged.append(cur)
                cur = g
        if cur:
            merged.append(cur)
        return merged

    def _len(self, s: str) -> int:
        return len(s)

    def _fixed_split(self, text: str) -> list[str]:
        if not text:
            return []
        size, ov = self.chunk_size, self.overlap
        step = max(1, size - ov)
        out: list[str] = []
        for start in range(0, len(text), step):
            out.append(text[start : start + size])
            if start + size >= len(text):
                break
        return out

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
        parts = self._split_text(text, self.separators)
        # Apply overlap by re-windowing merged parts (simple post-pass)
        if self.overlap <= 0:
            return [p for p in parts if p.strip()]
        combined = "\n\n".join(parts)
        return FixedChunker(self.chunk_size, self.overlap).chunk(combined)
