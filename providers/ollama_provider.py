"""Ollama local models via HTTP API."""

from __future__ import annotations

import base64
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from providers.base import BaseLLMProvider

DEFAULT_CAPTION_PROMPT = (
    """Describe this image in detail. Include objects, labels, arrows, relationships."""
)


class OllamaProvider(BaseLLMProvider):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        chat_model: str | None = None,
        vision_model: str | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._chat_model = chat_model or "llama3.2"
        self._vision_model = vision_model or "llava"

    def generate_response(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = False,
        temperature: float = 0.7,
    ) -> str | Iterator[str]:
        url = f"{self._base}/api/chat"
        payload: dict[str, Any] = {
            "model": self._chat_model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temperature},
        }
        if stream:

            def _gen() -> Iterator[str]:
                with httpx.Client(timeout=120.0) as client:
                    with client.stream("POST", url, json=payload) as r:
                        r.raise_for_status()
                        for line in r.iter_lines():
                            if not line:
                                continue
                            import json

                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            msg = obj.get("message") or {}
                            c = msg.get("content")
                            if c:
                                yield c
                            if obj.get("done"):
                                break

            return _gen()
        with httpx.Client(timeout=120.0) as client:
            r = client.post(url, json={**payload, "stream": False})
            r.raise_for_status()
            data = r.json()
        return (data.get("message") or {}).get("content", "").strip()

    def generate_caption(self, image_path: str, prompt: str | None = None) -> str:
        path = Path(image_path)
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": prompt or DEFAULT_CAPTION_PROMPT,
                "images": [b64],
            }
        ]
        url = f"{self._base}/api/chat"
        payload = {
            "model": self._vision_model,
            "messages": messages,
            "stream": False,
        }
        with httpx.Client(timeout=180.0) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        return (data.get("message") or {}).get("content", "").strip()
