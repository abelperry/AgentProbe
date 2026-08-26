"""HTTP-backed model client implementation."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from agent_probe.config import ModelConfig
from agent_probe.model_clients.base import BaseModelClient, ModelMessage, ModelResponse


class HttpModelClient(BaseModelClient):
    """Non-streaming HTTP client for OpenAI-compatible and Anthropic-compatible APIs."""

    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config

    def complete(
        self,
        messages: list[ModelMessage],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        if not self.model_config.api_key:
            raise ValueError("model api_key is empty")

        if self.model_config.format == "anthropic":
            return self._complete_anthropic(
                messages,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return self._complete_openai(
            messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _complete_openai(
        self,
        messages: list[ModelMessage],
        *,
        system: str | None,
        temperature: float,
        max_tokens: int | None,
    ) -> ModelResponse:
        payload_messages: list[dict[str, str]] = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend({"role": msg.role, "content": msg.content} for msg in messages)
        payload = {
            "model": self.model_config.model_name,
            "messages": payload_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.model_config.max_tokens,
        }
        raw = self._post_json(
            self._openai_chat_endpoint(self.model_config.base_url),
            payload,
            {
                "content-type": "application/json",
                "authorization": f"Bearer {self.model_config.api_key}",
            },
        )
        return ModelResponse(content=str(raw["choices"][0]["message"]["content"]).strip(), raw=raw)

    def _complete_anthropic(
        self,
        messages: list[ModelMessage],
        *,
        system: str | None,
        temperature: float,
        max_tokens: int | None,
    ) -> ModelResponse:
        system_parts = [system] if system else []
        payload_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                payload_messages.append({"role": msg.role, "content": msg.content})
        payload = {
            "model": self.model_config.model_name,
            "max_tokens": max_tokens or self.model_config.max_tokens,
            "temperature": temperature,
            "messages": payload_messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        raw = self._post_json(
            self._anthropic_messages_endpoint(self.model_config.base_url),
            payload,
            {
                "content-type": "application/json",
                "x-api-key": self.model_config.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        content = raw.get("content") or []
        texts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return ModelResponse(content="\n".join(texts).strip(), raw=raw)

    def _post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.model_config.timeout) as response:
                response_text = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body_text[:500]}") from exc
        return json.loads(response_text)

    @staticmethod
    def _openai_chat_endpoint(base_url: str) -> str:
        base = base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    @staticmethod
    def _anthropic_messages_endpoint(base_url: str) -> str:
        base = base_url.rstrip("/")
        if base.endswith("/v1/messages"):
            return base
        return f"{base}/v1/messages"
