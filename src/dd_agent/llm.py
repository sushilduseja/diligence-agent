"""Minimal Anthropic Messages API wrapper producing plain strings for the graph nodes."""

import httpx


class AnthropicLLM:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5", timeout: float = 60.0):
        self.api_key = api_key
        self.model = model
        self._client = httpx.Client(timeout=timeout)

    def invoke(self, prompt: str) -> str:
        resp = self._client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        text = "".join(block.get("text", "") for block in resp.json().get("content", []))
        return _strip_json_fence(text)


def _strip_json_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines and lines[0].startswith("```") else lines
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return text
