# src/llm.py
from __future__ import annotations
import os
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from tenacity import retry, wait_exponential_jitter, stop_after_attempt, retry_if_exception_type

def _get_api_key():
    k = os.getenv("OPENAI_API_KEY")
    if k:
        return k
    try:
        import streamlit as st
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None

class LLMClient:
    def __init__(self, settings):
        api_key = _get_api_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)
        self.settings = settings

    @retry(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
        wait=wait_exponential_jitter(initial=1, max=20),  # 1s, 2s, 4s … jittered
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _call(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=700,           # cap output size to reduce token usage
            timeout=60,               # seconds
        )
        return resp.choices[0].message.content

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self._call(system_prompt, user_prompt)
