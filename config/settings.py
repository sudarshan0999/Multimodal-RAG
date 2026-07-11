"""Application settings loaded from environment."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Env: EMBED_BACKEND, EMBED_MODEL_NAME, CHROMA_PERSIST_DIR, etc. (auto upper-case)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = ""
    groq_chat_model: str = "llama-3.3-70b-versatile"
    groq_vision_model: str = "llama-3.2-11b-vision-preview"

    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_vision_model: str = "gpt-4o-mini"

    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-02-15-preview"
    azure_openai_chat_deployment: str = ""
    azure_openai_vision_deployment: str = ""

    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.2"
    ollama_vision_model: str = "llava"

    embed_backend: Literal["sentence_transformers", "openai", "azure_openai"] = (
        "sentence_transformers"
    )
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    reranker_backend: Literal[
        "none", "ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-v2-m3"
    ] = "ms-marco-MiniLM-L-6-v2"

    chroma_persist_dir: Path = Field(default=Path("./chroma_data"))
    upload_dir: Path = Field(default=Path("./uploads"))
    captions_dir: Path = Field(default=Path("./captions"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
