"""OpenAI-compatible chat completions provider."""

from __future__ import annotations

import requests

from qkquant.ai.base import AiRequest, AiResponse
from qkquant.ai.prompt import SYSTEM_PROMPT, build_user_prompt


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout_seconds: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def analyze(self, request: AiRequest) -> AiResponse:
        if not self.api_key:
            return AiResponse(
                markdown="",
                provider=self.name,
                model=self.model,
                ok=False,
                error="missing API key",
            )
        if not self.base_url:
            return AiResponse(
                markdown="",
                provider=self.name,
                model=self.model,
                ok=False,
                error="missing base_url",
            )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(request)},
            ],
            "temperature": 0.2,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return AiResponse(
                markdown=str(content).strip(),
                provider=self.name,
                model=self.model,
            )
        except Exception as exc:
            return AiResponse(
                markdown="",
                provider=self.name,
                model=self.model,
                ok=False,
                error=str(exc),
            )


__all__ = ["OpenAICompatibleProvider"]
