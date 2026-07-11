"""Text embedding backends: local sentence-transformers, OpenAI, Azure OpenAI."""

from __future__ import annotations

from abc import ABC, abstractmethod
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from config.settings import Settings

# One SentenceTransformer per model per process — avoids re-download / re-init on every Streamlit rerun
# (repeated init can trigger huggingface_hub "Cannot send a request, as the client has been closed").
_st_model_cache: dict[str, Any] = {}
_st_model_lock = Lock()


def _get_sentence_transformer(model_name: str) -> Any:
    with _st_model_lock:
        if model_name not in _st_model_cache:
            from sentence_transformers import SentenceTransformer

            _st_model_cache[model_name] = SentenceTransformer(model_name)
        return _st_model_cache[model_name]


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        ...


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str) -> None:
        self._model = _get_sentence_transformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        emb = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [e.astype(float).tolist() for e in np.atleast_2d(emb)]

    def embed_query(self, text: str) -> list[float]:
        emb = self._model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return np.atleast_2d(emb)[0].astype(float).tolist()


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        batch = 64
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            resp = self._client.embeddings.create(model=self._model, input=chunk)
            by_idx = sorted(resp.data, key=lambda d: d.index)
            out.extend([list(d.embedding) for d in by_idx])
        return out

    def embed_query(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self._model, input=[text])
        return list(resp.data[0].embedding)


class AzureOpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        api_version: str,
        deployment: str,
    ) -> None:
        from openai import AzureOpenAI

        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self._deployment = deployment

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        batch = 64
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            resp = self._client.embeddings.create(
                model=self._deployment,
                input=chunk,
            )
            by_idx = sorted(resp.data, key=lambda d: d.index)
            out.extend([list(d.embedding) for d in by_idx])
        return out

    def embed_query(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(
            model=self._deployment,
            input=[text],
        )
        return list(resp.data[0].embedding)


def build_embedder(settings: Settings, override_backend: str | None = None) -> BaseEmbedder:
    backend = override_backend or settings.embed_backend
    if backend == "sentence_transformers":
        return SentenceTransformerEmbedder(settings.embed_model_name)
    if backend == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")
        return OpenAIEmbedder(settings.openai_api_key, settings.embed_model_name)
    if backend == "azure_openai":
        if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
            raise ValueError("Azure OpenAI key and endpoint required for Azure embeddings")
        dep = settings.embed_model_name
        return AzureOpenAIEmbedder(
            settings.azure_openai_api_key,
            settings.azure_openai_endpoint.rstrip("/"),
            settings.azure_openai_api_version,
            dep,
        )
    raise ValueError(f"Unknown embed backend: {backend}")


class Embedder:
    """Facade used by the app; delegates to BaseEmbedder."""

    def __init__(self, inner: BaseEmbedder) -> None:
        self._inner = inner

    @property
    def inner(self) -> BaseEmbedder:
        return self._inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)
