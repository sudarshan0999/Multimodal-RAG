"""OpenAI chat and vision."""

from __future__ import annotations

import base64
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from openai import OpenAI

from providers.base import BaseLLMProvider

DEFAULT_CAPTION_PROMPT = """Describe this image in detail for retrieval and Q&A.
Include: main objects, any visible text labels, arrows and relationships,
and diagram type. Be concise but complete."""


class OpenAIProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str,
        chat_model: str | None = None,
        vision_model: str | None = None,
    ) -> None:
        self._client = OpenAI(api_key=api_key)
        self._chat_model = chat_model or "gpt-4o-mini"
        self._vision_model = vision_model or "gpt-4o-mini"

    def _encode_image(self, image_path: str) -> str:
        path = Path(image_path)
        ext = path.suffix.lower().lstrip(".") or "png"
        if ext == "jpg":
            ext = "jpeg"
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:image/{ext};base64,{data}"

    def generate_response(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = False,
        temperature: float = 0.7,
    ) -> str | Iterator[str]:
        if stream:
            s = self._client.chat.completions.create(
                model=self._chat_model,
                messages=messages,
                stream=True,
                temperature=temperature,
            )

            def _gen() -> Iterator[str]:
                for chunk in s:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

            return _gen()
        resp = self._client.chat.completions.create(
            model=self._chat_model,
            messages=messages,
            stream=False,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    def generate_caption(self, image_path: str, prompt: str | None = None) -> str:
        uri = self._encode_image(image_path)
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or DEFAULT_CAPTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": uri}},
                ],
            }
        ]
        resp = self._client.chat.completions.create(
            model=self._vision_model,
            messages=messages,
            max_tokens=1024,
        )
        return (resp.choices[0].message.content or "").strip()
