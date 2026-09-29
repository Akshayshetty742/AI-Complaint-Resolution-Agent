import os
import json
import re
from typing import Optional, Dict, Any
from openai import OpenAI
from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, CATEGORIES, SENTIMENTS, PRIORITIES


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY", LLM_API_KEY)
        self.base_url = base_url or os.getenv("LLM_BASE_URL", LLM_BASE_URL)
        self.model = model or os.getenv("LLM_MODEL", LLM_MODEL)
        self._client: Optional[OpenAI] = None

    def is_configured(self) -> bool:
        """Returns True if a valid, non-placeholder API key is present."""
        if not self.api_key or not self.api_key.strip():
            return False
        if self.api_key.strip() in ["your_api_key_here", "sk-...", "your-key"]:
            return False
        return True

    def _get_client(self) -> OpenAI:
        if not self.is_configured():
            raise ValueError(
                "LLM API Key is missing or unconfigured!\n"
                "Please configure 'LLM_API_KEY' in your .env file or sidebar.\n"
                "You can also set LLM_BASE_URL and LLM_MODEL for any OpenAI-compatible provider."
            )
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 800) -> str:
        """Sends prompt to OpenAI-compatible LLM endpoint."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a precise, policy-grounded AI complaint resolution assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
        """Extracts JSON dictionary from raw LLM output, handling markdown code fences."""
        if not raw_text:
            return None

        # 1. Direct parse attempt
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # 2. Extract code block ```json ... ```
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        # 3. Extract any outer curly braces
        brace_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
        if brace_match:
            try:
                parsed = json.loads(brace_match.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        return None

    @staticmethod
    def validate_structured_output(data: Any) -> bool:
        """Validates that extracted JSON contains all required fields and acceptable values."""
        if not isinstance(data, dict):
            return False

        required_keys = ["category", "sentiment", "priority", "resolution", "customer_response"]
        for key in required_keys:
            if key not in data or not str(data[key]).strip():
                return False

        return True
