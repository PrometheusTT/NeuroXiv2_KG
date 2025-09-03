# -*- coding: utf-8 -*-
"""
A tiny LLM wrapper. If OPENAI_API_KEY is empty, all calls become no-ops and return None.
This avoids "Illegal header value b'Bearer '" errors.
"""
from typing import Optional, Dict, Any

class LLM:
    def __init__(self, openai_api_key: str = "", model: str = "gpt-4o"):
        self.api_key = (openai_api_key or "").strip()
        self.model = model

    def chat(self, messages: list, response_format: Optional[Dict[str, Any]] = None, temperature: Optional[float] = None) -> Optional[str]:
        if not self.api_key:
            return None  # offline fallback
        try:
            import httpx
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {"model": self.model, "messages": messages}
            # Some hosted models reject non-default temperature; avoid unless explicitly set.
            if temperature is not None:
                payload["temperature"] = temperature
            # response_format is not always supported; skip by default
            r = httpx.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception:
            return None
