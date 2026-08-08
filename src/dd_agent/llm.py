"""Minimal Groq Chat Completions wrapper producing plain strings for the graph nodes."""

import httpx


class GroqLLM:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", timeout: float = 60.0):
        self.api_key = api_key
        self.model = model
        self._client = httpx.Client(timeout=timeout)

    def invoke(self, prompt: str) -> str:
        resp = self._client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return _strip_json_fence(text)


def _strip_json_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines and lines[0].startswith("```") else lines
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return text
