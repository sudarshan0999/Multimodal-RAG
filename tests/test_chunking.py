from chunking.fixed import FixedChunker
from chunking.recursive import RecursiveChunker
from chunking.token import TokenChunker


def test_fixed_chunker_overlap():
    c = FixedChunker(chunk_size=20, overlap=5)
    text = "a" * 50
    parts = c.chunk(text)
    assert len(parts) >= 2
    assert all(len(p) <= 20 for p in parts)


def test_recursive_non_empty():
    c = RecursiveChunker(chunk_size=100, overlap=10)
    text = "Paragraph one.\n\nParagraph two is here."
    parts = c.chunk(text)
    assert parts
    assert "".join(parts).replace("\n", "").replace(" ", "") in text.replace("\n", "").replace(
        " ", ""
    ) or set("".join(parts)).issubset(set(text))


def test_token_chunker():
    c = TokenChunker(chunk_size=32, overlap=4)
    text = "Hello world. " * 20
    parts = c.chunk(text)
    assert parts
