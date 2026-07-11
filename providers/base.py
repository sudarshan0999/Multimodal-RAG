from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class BaseLLMProvider(ABC):
    """Pluggable LLM for chat and image captioning."""

    @abstractmethod
    def generate_response(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = False,
        temperature: float = 0.7,
    ) -> str | Iterator[str]:
        """Chat completion. Messages use OpenAI-style role/content."""

    @abstractmethod
    def generate_caption(self, image_path: str, prompt: str | None = None) -> str:
        """Describe image at path (diagrams, labels, arrows, relationships)."""
