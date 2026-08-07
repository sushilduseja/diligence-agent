import json

import pytest

from dd_agent.llm import _strip_json_fence


def test_strip_json_fence_removes_markdown_fence():
    raw = '```json\n{"a": 1}\n```'
    assert _strip_json_fence(raw) == '{"a": 1}'


def test_strip_json_fence_passthrough():
    assert _strip_json_fence('{"a": 1}') == '{"a": 1}'


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeHTTPX:
    def __init__(self, responses):
        self.responses = list(responses)
        self.last_kwargs = None

    def post(self, url, **kwargs):
        self.last_kwargs = kwargs
        return self.responses.pop(0)


def test_invoke_extracts_text_and_strips_fence(monkeypatch):
    from dd_agent import llm

    fake = FakeHTTPX(
        [FakeResponse({"content": [{"type": "text", "text": '```json\n{"x": 1}\n```'}]})]
    )
    monkeypatch.setattr(llm.httpx, "Client", lambda **kw: fake)
    model = llm.AnthropicLLM("sk-test", model="claude-test")
    result = model.invoke("prompt")
    assert result == '{"x": 1}'
    assert fake.last_kwargs["json"]["model"] == "claude-test"
    assert fake.last_kwargs["headers"]["x-api-key"] == "sk-test"
