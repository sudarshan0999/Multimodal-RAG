import fitz

from retrieval.hybrid import HybridRetriever
from vectorstore.chroma_store import ChromaStore
from vectorstore.embedder import BaseEmbedder, Embedder


class FakeEmbedder(BaseEmbedder):
    """Deterministic pseudo-embeddings for tests (no model download)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            v = [0.0] * 8
            for i, ch in enumerate(t[:8].lower()):
                v[i] = (ord(ch) % 13) / 13.0
            out.append(v)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _tiny_pdf(path: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "RAG test document about electric circuits.")
    doc.save(path)
    doc.close()


def test_chroma_ingest_and_hybrid(tmp_path):
    pdf = tmp_path / "t.pdf"
    _tiny_pdf(str(pdf))

    emb = Embedder(FakeEmbedder())
    store = ChromaStore(tmp_path / "chroma", embedder=emb)
    store.add_documents(
        documents=["electric circuits use resistors", "diagram shows arrows between blocks"],
        metadatas=[
            {"source": "t.pdf", "source_hash": "abc", "page": 1, "type": "text", "image_path": ""},
            {"source": "t.pdf", "source_hash": "abc", "page": 1, "type": "image", "image_path": "/x.png"},
        ],
    )

    hr = HybridRetriever(store, semantic_weight=0.6, bm25_weight=0.4)
    out = hr.retrieve("electric circuit", top_k=2, fetch_k=10)
    assert out
    assert any("electric" in (o.document or "").lower() for o in out)
