from __future__ import annotations

from typing import Literal

from config.settings import Settings
from providers.azure_openai_provider import AzureOpenAIProvider
from providers.base import BaseLLMProvider
from providers.groq_provider import GroqProvider
from providers.ollama_provider import OllamaProvider
from providers.openai_provider import OpenAIProvider

ProviderKind = Literal["groq", "openai", "azure_openai", "ollama"]


def get_llm_provider(
    kind: ProviderKind,
    settings: Settings,
    *,
    api_key_override: str | None = None,
) -> BaseLLMProvider:
    if kind == "groq":
        key = api_key_override or settings.groq_api_key
        if not key:
            raise ValueError("GROQ_API_KEY is required for Groq provider")
        return GroqProvider(
            key,
            chat_model=settings.groq_chat_model,
            vision_model=settings.groq_vision_model,
        )
    if kind == "openai":
        key = api_key_override or settings.openai_api_key
        if not key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
        return OpenAIProvider(
            key,
            chat_model=settings.openai_chat_model,
            vision_model=settings.openai_vision_model,
        )
    if kind == "azure_openai":
        key = api_key_override or settings.azure_openai_api_key
        if not key or not settings.azure_openai_endpoint:
            raise ValueError("Azure OpenAI key and endpoint required")
        if not settings.azure_openai_chat_deployment or not settings.azure_openai_vision_deployment:
            raise ValueError("Azure chat and vision deployment names required")
        return AzureOpenAIProvider(
            key,
            settings.azure_openai_endpoint,
            settings.azure_openai_api_version,
            settings.azure_openai_chat_deployment,
            settings.azure_openai_vision_deployment,
        )
    if kind == "ollama":
        return OllamaProvider(
            settings.ollama_base_url,
            chat_model=settings.ollama_chat_model,
            vision_model=settings.ollama_vision_model,
        )
    raise ValueError(f"Unknown provider: {kind}")
