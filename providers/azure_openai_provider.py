"""Azure OpenAI chat and vision."""

from __future__ import annotations

import base64
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from openai import AzureOpenAI

from providers.base import BaseLLMProvider

DEFAULT_CAPTION_PROMPT = """Describe this image in detail for retrieval and Q&A.
Include: main objects, labels, arrows, relationships. Be concise but complete."""


class AzureOpenAIProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        api_version: str,
        chat_deployment: str,
        vision_deployment: str,
    ) -> None:
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint.rstrip("/"),
            api_version=api_version,
        )
        self._chat_deployment = chat_deployment
        self._vision_deployment = vision_deployment

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
                model=self._chat_deployment,
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
            model=self._chat_deployment,
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
            model=self._vision_deployment,
            messages=messages,
            max_tokens=1024,
        )
        return (resp.choices[0].message.content or "").strip()
