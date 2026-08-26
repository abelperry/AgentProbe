"""Tests for reusable model API clients."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from agent_probe.config import ModelConfig
from agent_probe.model_clients import HttpModelClient, ModelMessage, create_model_client


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_create_model_client_returns_http_client() -> None:
    client = create_model_client(
        ModelConfig(base_url="https://example.test/v1", api_key="k", model_name="m")
    )

    assert isinstance(client, HttpModelClient)


def test_http_model_client_openai_payload(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> _FakeResponse:
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = HttpModelClient(
        ModelConfig(
            base_url="https://example.test/v1",
            api_key="k",
            model_name="m",
            timeout=7,
            format="openai",
        )
    )
    response = client.complete([ModelMessage(role="user", content="hi")], system="sys")

    assert response.content == "ok"
    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert seen["timeout"] == 7
    assert seen["payload"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def test_http_model_client_anthropic_payload(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> _FakeResponse:
        seen["url"] = request.full_url
        seen["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"content": [{"type": "text", "text": "ok"}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = HttpModelClient(
        ModelConfig(
            base_url="https://example.test/anthropic",
            api_key="k",
            model_name="m",
            format="anthropic",
        )
    )
    response = client.complete([ModelMessage(role="user", content="hi")], system="sys")

    assert response.content == "ok"
    assert seen["url"] == "https://example.test/anthropic/v1/messages"
    assert seen["payload"]["system"] == "sys"
    assert seen["payload"]["messages"] == [{"role": "user", "content": "hi"}]
