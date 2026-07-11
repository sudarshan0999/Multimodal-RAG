"""
Multimodal RAG — Streamlit UI.
Run from project root: streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project root on path when running streamlit from repo root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from chunking.factory import get_chunker
from config.settings import Settings, get_settings
from ingestion.pipeline import IngestionPipeline
from providers.factory import ProviderKind, get_llm_provider
from retrieval.hybrid import HybridRetriever
from retrieval.rag import RAGPipeline, SYSTEM_PROMPT, build_user_prompt
from retrieval.reranker import get_reranker, RerankerKind
from vectorstore.chroma_store import ChromaStore
from vectorstore.embedder import Embedder, build_embedder


def _session_defaults() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_retrieval_debug" not in st.session_state:
        st.session_state.last_retrieval_debug = None


def _sidebar_settings(settings: Settings) -> dict:
    st.sidebar.header("Settings")
    provider: ProviderKind = st.sidebar.selectbox(
        "LLM provider",
        ["groq", "openai", "azure_openai", "ollama"],
        index=0,
        format_func=lambda x: x.replace("_", " ").title(),
    )
    use_env = st.sidebar.checkbox("Load API keys from .env", value=True)

    api_key = ""
    if provider == "groq":
        api_key = st.sidebar.text_input("Groq API key", type="password", value="")
    elif provider == "openai":
        api_key = st.sidebar.text_input("OpenAI API key", type="password", value="")
    elif provider == "azure_openai":
        api_key = st.sidebar.text_input(
            "Azure OpenAI API key", type="password", value=""
        )

    st.sidebar.subheader("Chunking")
    chunk_strategy = st.sidebar.selectbox(
        "Strategy",
        ["fixed", "recursive", "token", "semantic"],
    )
    chunk_size = st.sidebar.number_input(
        "Chunk size", min_value=64, max_value=8000, value=1000
    )
    overlap = st.sidebar.number_input("Overlap", min_value=0, max_value=2000, value=200)

    st.sidebar.subheader("Retrieval")
    top_k = st.sidebar.number_input(
        "Top-k",
        min_value=1,
        max_value=50,
        value=1,
        help="Number of pages to fetch - 1 = single page for accuracy",
    )
    fetch_k = st.sidebar.number_input(
        "Semantic pool (fetch_k)",
        min_value=5,
        max_value=200,
        value=10,
        help="Initial candidate pool",
    )
    image_top_k = st.sidebar.number_input(
        "Extra images per query",
        min_value=0,
        max_value=12,
        value=3,
        help="Additional diagrams to include",
    )
    sem_w = st.sidebar.slider(
        "Semantic weight",
        0.0,
        1.0,
        0.5,
        0.05,
        help="0.5 = balanced, higher = semantic similarity, lower = keyword match",
    )
    bm25_w = st.sidebar.slider(
        "BM25 weight",
        0.0,
        1.0,
        0.5,
        0.05,
        help="0.5 = balanced, higher = keyword matching",
    )

    st.sidebar.subheader("Embeddings")
    embed_backend = st.sidebar.selectbox(
        "Embed backend",
        ["sentence_transformers", "openai", "azure_openai"],
        index=0,
    )
    embed_model = st.sidebar.text_input(
        "Embed model / deployment",
        value=settings.embed_model_name,
    )

    st.sidebar.subheader("Reranking")
    reranker_kind: RerankerKind = st.sidebar.selectbox(
        "Reranker",
        ["none", "ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-v2-m3"],
        index=1,
        format_func=lambda x: x if x == "none" else x.replace("_", " ").title(),
        help="none: no reranking, ms-marco-MiniLM-L-6-v2: fast cross-encoder (default), BAAI/bge-reranker-v2-m3: more accurate but slower",
    )
    show_rerank_debug = st.sidebar.checkbox(
        "Show reranker debug",
        value=True,
        help="Show before/after ranking comparison in debug panel",
    )

    st.sidebar.subheader("Generation")
    stream_ans = st.sidebar.checkbox("Stream responses", value=True)
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.3,
        step=0.1,
        help="Lower = more focused, Higher = more creative",
    )
    force_reindex = st.sidebar.checkbox("Force re-index on next upload", value=False)
    clear_all_data = st.sidebar.checkbox(
        "Clear all stored data",
        value=False,
        help="Delete all indexed documents, uploaded files, and cached data from ChromaDB and upload folders",
    )

    if clear_all_data:
        st.sidebar.warning("⚠️ This will DELETE all data!")
        confirm_clear = st.sidebar.checkbox(
            "I understand - clear everything", value=False
        )
    else:
        confirm_clear = False

    return {
        "provider": provider,
        "use_env": use_env,
        "api_key": api_key.strip(),
        "chunk_strategy": chunk_strategy,
        "chunk_size": int(chunk_size),
        "overlap": int(overlap),
        "top_k": int(top_k),
        "fetch_k": int(fetch_k),
        "image_top_k": int(image_top_k),
        "sem_w": float(sem_w),
        "bm25_w": float(bm25_w),
        "embed_backend": embed_backend,
        "embed_model": embed_model.strip(),
        "reranker_kind": reranker_kind,
        "show_rerank_debug": show_rerank_debug,
        "stream": stream_ans,
        "temperature": float(temperature),
        "force_reindex": force_reindex,
        "clear_all_data": clear_all_data and confirm_clear,
    }


def _effective_settings(base: Settings, cfg: dict) -> Settings:
    data = base.model_dump()
    data["embed_backend"] = cfg["embed_backend"]
    data["embed_model_name"] = cfg["embed_model"] or base.embed_model_name
    return Settings(**data)


def _get_llm(cfg: dict, settings: Settings):
    key = None if cfg["use_env"] else cfg["api_key"]
    if cfg["use_env"]:
        return get_llm_provider(cfg["provider"], settings, api_key_override=None)
    if cfg["api_key"]:
        return get_llm_provider(
            cfg["provider"], settings, api_key_override=cfg["api_key"]
        )
    return get_llm_provider(cfg["provider"], settings, api_key_override=None)


def main() -> None:
    st.set_page_config(page_title="Multimodal RAG", layout="wide")
    _session_defaults()
    settings = get_settings()
    cfg = _sidebar_settings(settings)
    eff = _effective_settings(settings, cfg)

    st.title("Multimodal RAG")
    st.caption(
        "PDFs → text + diagram extraction → captions → ChromaDB → hybrid retrieval"
    )

    persist = eff.chroma_persist_dir
    persist.mkdir(parents=True, exist_ok=True)
    eff.upload_dir.mkdir(parents=True, exist_ok=True)
    eff.captions_dir.mkdir(parents=True, exist_ok=True)

    if cfg.get("clear_all_data"):
        import shutil
        import time
        import os
        import stat

        with st.spinner("Clearing all stored data..."):
            import gc

            gc.collect()
            time.sleep(1)

            def handle_remove_readonly(func, path, exc):
                os.chmod(path, stat.S_IWRITE)
                func(path)

            try:
                if persist.exists():
                    shutil.rmtree(persist, onerror=handle_remove_readonly)
                    st.success("✅ Cleared ChromaDB vector store")
            except Exception as e:
                st.warning(
                    f"Could not clear ChromaDB: {e}. Try closing any apps using it first."
                )
            try:
                if eff.upload_dir.exists():
                    shutil.rmtree(eff.upload_dir, onerror=handle_remove_readonly)
                    st.success("✅ Cleared uploaded files")
            except Exception:
                pass
            try:
                if eff.captions_dir.exists():
                    shutil.rmtree(eff.captions_dir, onerror=handle_remove_readonly)
                    st.success("✅ Cleared cached captions")
            except Exception:
                pass
            st.rerun()

    try:
        inner = build_embedder(eff, override_backend=cfg["embed_backend"])
        embedder = Embedder(inner)
    except Exception as e:
        st.error(f"Embedder init failed: {e}")
        st.stop()

    store = ChromaStore(persist, embedder=embedder)

    try:
        llm = _get_llm(cfg, eff)
    except Exception as e:
        st.warning(f"LLM provider not ready: {e}. Set keys in sidebar or .env.")
        llm = None

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded and llm is not None:
        save_path = eff.upload_dir / uploaded.name
        if st.button("Ingest PDF", type="primary"):
            with st.status("Processing PDF…", expanded=True) as status:
                save_path.write_bytes(uploaded.getvalue())
                status.update(label="Extracting text and diagrams…", state="running")

                chunker = get_chunker(
                    cfg["chunk_strategy"],  # type: ignore[arg-type]
                    cfg["chunk_size"],
                    cfg["overlap"],
                    semantic_embedder=embedder.inner
                    if cfg["chunk_strategy"] == "semantic"
                    else None,
                )

                pipe = IngestionPipeline(eff, store, llm, embedder=embedder)

                def on_status(msg: str) -> None:
                    status.update(label=msg, state="running")

                try:
                    result = pipe.ingest_pdf(
                        save_path,
                        chunker=chunker,
                        force_reindex=cfg["force_reindex"],
                        embed_backend=cfg["embed_backend"],
                        on_status=on_status,
                    )
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
                    status.update(label="Failed", state="error")
                else:
                    if result.skipped:
                        status.update(
                            label="Skipped (already indexed)", state="complete"
                        )
                    else:
                        status.update(
                            label=f"Done: {result.text_chunks} text chunks, {result.image_records} images",
                            state="complete",
                        )
                    st.success(
                        f"Source `{result.source}` — text chunks: {result.text_chunks}, "
                        f"images: {result.image_records}, skipped: {result.skipped}"
                    )

    st.divider()
    st.subheader("Chat")

    if llm is None:
        st.info("Configure an LLM provider to chat and ingest.")
        return

    reranker = None
    # if cfg["reranker_kind"] != "none":
    #     try:
    #         reranker = get_reranker(cfg["reranker_kind"])
    #     except Exception as e:
    #         st.warning(f"Reranker unavailable: {e}")

    retriever = HybridRetriever(
        store,
        semantic_weight=cfg["sem_w"],
        bm25_weight=cfg["bm25_w"],
        image_top_k=cfg["image_top_k"],
        reranker=reranker,
    )

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask about your documents…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                retrieved = retriever.retrieve(
                    prompt,
                    top_k=cfg["top_k"],
                    fetch_k=cfg["fetch_k"],
                )
                st.session_state.last_retrieval_debug = retrieved

                reranker_active = cfg.get("reranker_kind", "none") != "none"

                if reranker_active:
                    st.caption(
                        f"Query: '{prompt}' | Retrieved {len(retrieved)} results | Reranker: {cfg['reranker_kind']} ACTIVE"
                    )
                else:
                    st.caption(
                        f"Query: '{prompt}' | Retrieved {len(retrieved)} results | Reranker: OFF"
                    )

                all_page_content = retriever.get_all_by_page(retrieved)

                combined_context = retrieved + all_page_content

                user_content = build_user_prompt(prompt, combined_context, None)

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]

                temperature = cfg.get("temperature", 0.3)
                if cfg["stream"]:
                    acc: list[str] = []

                    def _stream():
                        for t in llm.generate_response(
                            messages, stream=True, temperature=temperature
                        ):
                            acc.append(t)
                            yield t

                    st.write_stream(_stream())
                    answer = "".join(acc)
                else:
                    resp = llm.generate_response(
                        messages, stream=False, temperature=temperature
                    )
                    answer = resp if isinstance(resp, str) else ""
                    st.markdown(answer)

                paths = RAGPipeline.image_paths_from_retrieval(combined_context)
                if paths:
                    st.markdown("**Figures/Diagrams from matched pages:**")
                    ncols = min(4, max(1, len(paths)))
                    cols = st.columns(ncols)
                    for i, p in enumerate(paths[:12]):
                        with cols[i % ncols]:
                            st.image(p, use_container_width=True)

                table_data = [
                    (r.metadata, r.document)
                    for r in combined_context
                    if r.metadata.get("type") == "table"
                ]
                if table_data:
                    st.markdown("**Tables from your query pages:**")
                    for meta, doc in table_data[:5]:
                        with st.expander(f"Table on page {meta.get('page', '?')}"):
                            st.caption(doc)
                            cols = meta.get("columns", [])
                            if cols:
                                st.caption(f"Columns: {', '.join(cols)}")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as e:
                error_msg = str(e)
                if (
                    "image" in error_msg.lower()
                    and "does not support" in error_msg.lower()
                ):
                    st.error(
                        "Image captioning failed: The selected vision model does not support image input."
                    )
                    st.info(
                        "Fix: Change GROQ_VISION_MODEL to a vision-capable model (e.g., llama-3.2-11b-vision-preview) in .env file"
                    )
                elif "caption" in error_msg.lower() and "400" in error_msg:
                    st.error(
                        "Caption API error (400): Check vision model configuration."
                    )
                    st.info("Ensure GROQ_VISION_MODEL is set to a vision model in .env")
                else:
                    st.error(error_msg)

            with st.expander("Debug: Retrieval + Full Context", expanded=False):
                dbg = st.session_state.get("last_retrieval_debug") or []
                reranker_kind = cfg.get("reranker_kind", "none")

                st.markdown("#### 1. Reranker Status")
                if reranker_kind != "none":
                    st.success(f"Reranker ACTIVE: {reranker_kind}")
                else:
                    st.warning("Reranker: OFF")

                st.markdown("#### 2. Retrieved Documents")
                if not dbg:
                    st.write("No retrieval yet.")
                else:
                    st.caption(f"Showing {len(dbg)} retrieved results")
                    for idx, r in enumerate(dbg, start=1):
                        meta = r.metadata
                        page = meta.get("page", "?")
                        typ = meta.get("type", "text")
                        src = meta.get("source", "")
                        chunk_idx = meta.get("chunk_index", "?")
                        score = r.fused_score
                        preview = (r.document or "")[:100].replace("\n", " ")
                        st.markdown(
                            f"**{idx}.** [{typ}] p{page} | chunk:{chunk_idx} | score:{score:.3f}"
                        )
                        st.caption(f"{preview}...")

                st.markdown("#### 3. All Content in Context (text + table + image)")
                all_docs = dbg or []
                text_count = sum(
                    1 for r in all_docs if r.metadata.get("type") == "text"
                )
                table_count = sum(
                    1 for r in all_docs if r.metadata.get("type") == "table"
                )
                img_count = sum(
                    1
                    for r in all_docs
                    if r.metadata.get("type") in ["image", "vector_graphic"]
                )
                pages = sorted(set(r.metadata.get("page", "?") for r in all_docs))

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Text", text_count)
                col2.metric("Tables", table_count)
                col3.metric("Images", img_count)
                col4.metric(
                    "Pages",
                    ", ".join(str(p) for p in pages[:5])
                    + ("..." if len(pages) > 5 else ""),
                )

                with st.expander("View All Content Details"):
                    for i, r in enumerate(all_docs, 1):
                        meta = r.metadata
                        typ = meta.get("type", "text")
                        pg = meta.get("page", "?")
                        src = meta.get("source", "")

                        with st.expander(
                            f"{i}. [{typ.upper()}] Page {pg} | Source: {src}"
                        ):
                            st.json(meta)
                            if typ == "text":
                                st.code((r.document or "")[:300])
                            elif typ == "table":
                                st.caption(r.document or "")
                            elif typ in ["image", "vector_graphic"]:
                                ip = meta.get("image_path", "")
                                st.caption(f"File: {ip}")


if __name__ == "__main__":
    main()
