"""Ingestion cache keys for skipping re-embed."""

from __future__ import annotations

import hashlib


def ingestion_cache_key(
    file_bytes: bytes,
    chunk_strategy: str,
    chunk_size: int,
    overlap: int,
    embed_model_name: str,
) -> str:
    h = hashlib.sha256()
    h.update(file_bytes)
    h.update(f"|{chunk_strategy}|{chunk_size}|{overlap}|{embed_model_name}".encode())
    return h.hexdigest()


def source_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def should_skip_ingest(
    store_has_source_hash: bool,
    force_reindex: bool,
) -> bool:
    return store_has_source_hash and not force_reindex
