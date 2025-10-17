from __future__ import annotations
from typing import Optional, List, Dict, Any
from .config import Settings
import os

class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("Install openai>=1.0.0 to use LLM") from e
        kwargs = {}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self.client = OpenAI(api_key=settings.openai_api_key, **kwargs)

    def generate(self, system: str, user: str, max_tokens: int = 800) -> str:
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        resp = self.client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
