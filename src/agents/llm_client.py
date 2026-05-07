import os
import re
import json
import time
import random
from typing import Any, Optional

import ollama
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    genai.configure(api_key=gemini_key)


class LLMClient:
    """
    Shared LLM client for Ollama, Gemini, and future Claude support.
    """

    def __init__(
        self,
        model_type: str = "ollama",
        model_name: str = "qwen2.5",
        temperature: float = 0.0,
        max_retries: int = 3,
        base_delay: float = 1.0,
        **default_options: Any,
    ):
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.default_options = default_options

    def _sleep_before_retry(self, attempt: int):
        delay = self.base_delay * (2 ** attempt)
        delay += random.uniform(0, 0.5)
        time.sleep(delay)

    def _merged_options(self, **overrides: Any) -> dict:
        options = {
            "temperature": self.temperature,
            **self.default_options,
            **overrides,
        }
        return {k: v for k, v in options.items() if v is not None}

    def call_text(
        self,
        prompt: str,
        system: str = "Follow formatting strictly.",
        **options: Any,
    ) -> str:
        """
        Call the selected model and return plain text.
        """

        last_error = None

        for attempt in range(self.max_retries):
            try:
                if self.model_type == "ollama":
                    response = ollama.chat(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        options=self._merged_options(**options),
                    )
                    return response["message"]["content"].strip()

                elif self.model_type == "gemini":
                    model = genai.GenerativeModel(self.model_name)

                    generation_config = {
                        "temperature": self.temperature,
                        **self.default_options,
                        **options,
                    }

                    response = model.generate_content(
                        prompt,
                        generation_config=generation_config,
                    )

                    return response.text.strip()

                elif self.model_type == "claude":
                    return self._call_claude_text(
                        prompt=prompt,
                        system=system,
                        **options,
                    )

                else:
                    raise ValueError(f"Unsupported model type: {self.model_type}")

            except Exception as e:
                last_error = e
                print(f"⚠️ LLM call failed on attempt {attempt + 1}/{self.max_retries}: {e}")

                if attempt < self.max_retries - 1:
                    self._sleep_before_retry(attempt)

        print(f"❌ LLM failed after {self.max_retries} attempts: {last_error}")
        return ""

    def call_json(
        self,
        prompt: str,
        system: str = "Return ONLY valid JSON.",
        **options: Any,
    ) -> dict:
        """
        Call the selected model and parse a JSON object from the response.
        """

        raw = self.call_text(prompt, system=system, **options)

        if not raw:
            return {}

        try:
            return json.loads(raw)
        except Exception:
            pass

        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return {}
            return json.loads(match.group(0))
        except Exception as e:
            print(f"❌ JSON parse error: {e}")
            print(f"Raw output was: {raw}")
            return {}

    def _call_claude_text(
        self,
        prompt: str,
        system: str = "Follow formatting strictly.",
        **options: Any,
    ) -> str:
        """
        Optional Claude support.

        Requires:
            pip install anthropic

        Env:
            ANTHROPIC_API_KEY=...
        """

        try:
            import anthropic
        except ImportError:
            raise ImportError("Install Claude support with: pip install anthropic")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is missing.")

        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model=self.model_name,
            max_tokens=options.pop("max_tokens", 1000),
            temperature=options.pop("temperature", self.temperature),
            system=system,
            messages=[
                {"role": "user", "content": prompt}
            ],
            **options,
        )

        chunks = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                chunks.append(block.text)

        return "\n".join(chunks).strip()