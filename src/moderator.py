from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import (
    MODERATION_MAX_TOKENS,
    MODERATION_MODEL_NAME,
    MODERATION_USE_JSON_MODE,
    MODERATOR_PROMPT_PATH,
    ROOT_DIR,
)


@dataclass(frozen=True)
class ModerationDecision:
    """Structured decision returned by the moderation agent."""

    is_prompt_injection: bool
    reason: str
    category: str | None = None
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_prompt_injection": self.is_prompt_injection,
            "reason": self.reason,
            "category": self.category,
            "raw_response": self.raw_response,
        }


class BaseAgent(ABC):
    """Generic agent contract used to keep the architecture explicit."""

    name: str

    @abstractmethod
    def run(self, input_text: str) -> dict[str, Any]:
        raise NotImplementedError


class BaseModeratorAgent(BaseAgent):
    """Contract consumed by Adrien before calling the main RAG LLM."""

    @abstractmethod
    def moderate(self, question: str) -> dict[str, Any]:
        raise NotImplementedError

    def run(self, input_text: str) -> dict[str, Any]:
        return self.moderate(input_text)


class PromptInjectionAgent(BaseModeratorAgent):
    """Base class for prompt-injection moderation agents."""

    name = "prompt_injection_moderator"

    def __init__(self, fail_closed: bool = True) -> None:
        self.fail_closed = fail_closed

    def moderate(self, question: str) -> dict[str, Any]:
        return self.moderate_decision(question).to_dict()

    @abstractmethod
    def moderate_decision(self, question: str) -> ModerationDecision:
        raise NotImplementedError

    def _parse_decision(self, content: str) -> ModerationDecision:
        parsed = self._parse_json_object(content)
        if parsed is not None:
            return self._decision_from_json(parsed=parsed, raw_response=content)

        normalized = content.strip().lower()
        if normalized.startswith("unsafe"):
            return ModerationDecision(
                is_prompt_injection=True,
                category="unsafe",
                reason="The moderation agent classified the message as unsafe.",
                raw_response=content,
            )
        if normalized.startswith("safe"):
            return ModerationDecision(
                is_prompt_injection=False,
                category="safe",
                reason="The moderation agent classified the message as safe.",
                raw_response=content,
            )

        return ModerationDecision(
            is_prompt_injection=self.fail_closed,
            category="parse_error",
            reason="The moderation response was not valid JSON; fail-closed policy applied."
            if self.fail_closed
            else "The moderation response was not valid JSON; fail-open policy applied.",
            raw_response=content,
        )

    def _decision_from_json(
        self,
        parsed: dict[str, Any],
        raw_response: str,
    ) -> ModerationDecision:
        if "is_prompt_injection" in parsed:
            is_prompt_injection = self._coerce_bool(parsed["is_prompt_injection"])
        elif "violation" in parsed:
            is_prompt_injection = self._coerce_bool(parsed["violation"])
        else:
            is_prompt_injection = self.fail_closed

        reason = parsed.get("reason") or parsed.get("rationale") or "No reason provided."
        category = parsed.get("category")

        return ModerationDecision(
            is_prompt_injection=is_prompt_injection,
            reason=str(reason),
            category=str(category) if category is not None else None,
            raw_response=raw_response,
        )

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float):
            return value != 0
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "oui", "violation", "unsafe"}:
                return True
            if normalized in {"false", "0", "no", "non", "safe"}:
                return False
        return bool(value)

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            return None

        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


class GroqModeratorAgent(PromptInjectionAgent):
    """Groq-backed agent dedicated to prompt-injection detection.

    This agent is intentionally separate from the main RAG generation model.
    Adrien should call it before retrieval/generation and stop the pipeline when
    `is_prompt_injection` is true.
    """

    provider = "groq"

    def __init__(
        self,
        client: Any | None = None,
        model_name: str = MODERATION_MODEL_NAME,
        prompt_path: str | Path = MODERATOR_PROMPT_PATH,
        use_json_mode: bool = MODERATION_USE_JSON_MODE,
        max_tokens: int = MODERATION_MAX_TOKENS,
        fail_closed: bool = True,
    ) -> None:
        super().__init__(fail_closed=fail_closed)
        self.client = client or self._build_default_client()
        self.model_name = model_name
        self.prompt_path = Path(prompt_path)
        self.use_json_mode = use_json_mode
        self.max_tokens = max_tokens

    def moderate_decision(self, question: str) -> ModerationDecision:
        if not question or not question.strip():
            raise ValueError("Question must be non-empty.")

        response = self.client.chat.completions.create(**self._build_request(question.strip()))
        content = response.choices[0].message.content or ""
        return self._parse_decision(content)

    def _build_request(self, question: str) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self._load_prompt()},
                {"role": "user", "content": question},
            ],
            "temperature": 0,
            "max_completion_tokens": self.max_tokens,
        }
        if self.use_json_mode:
            request["response_format"] = {"type": "json_object"}
        return request

    def _load_prompt(self) -> str:
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Moderator prompt not found: {self.prompt_path}")
        return self.prompt_path.read_text(encoding="utf-8")

    @staticmethod
    def _build_default_client() -> Any:
        from dotenv import load_dotenv
        from groq import Groq

        load_dotenv(ROOT_DIR / ".env")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
        return Groq(api_key=api_key)


# Short aliases kept for simple imports in the RAG pipeline.
ModeratorAgent = GroqModeratorAgent
Moderator = GroqModeratorAgent
