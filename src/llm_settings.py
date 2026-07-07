from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_GENERATION_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GENERATION_FALLBACK_MODELS = (
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
)

DEFAULT_MODERATION_MODEL = "openai/gpt-oss-safeguard-20b"
DEFAULT_MODERATION_FALLBACK_MODELS = (
    "meta-llama/llama-prompt-guard-2-86m",
    "meta-llama/llama-prompt-guard-2-22m",
)

API_KEY_ENV_NAMES = (
    "GROQ_API_KEY",
    "GROQ_API_KEY_BACKUP",
    "GROQ_API_KEY_FALLBACK",
)

PLACEHOLDER_KEY_PARTS = (
    "replace",
    "your_groq_api_key",
    "mettre",
    "api_key",
    "xxx",
)


@dataclass(frozen=True)
class LLMSettings:
    groq_api_key: str | None
    api_key_source: str | None
    generation_model: str
    generation_fallback_models: tuple[str, ...]
    moderation_model: str
    moderation_fallback_models: tuple[str, ...]
    timeout_seconds: float

    @property
    def generation_model_chain(self) -> tuple[str, ...]:
        return unique_model_chain(
            self.generation_model,
            self.generation_fallback_models,
        )

    @property
    def moderation_model_chain(self) -> tuple[str, ...]:
        return unique_model_chain(
            self.moderation_model,
            self.moderation_fallback_models,
        )


def load_llm_settings() -> LLMSettings:
    load_dotenv_if_available()
    api_key, api_key_source = first_available_api_key()

    return LLMSettings(
        groq_api_key=api_key,
        api_key_source=api_key_source,
        generation_model=os.getenv("GROQ_GENERATION_MODEL", DEFAULT_GENERATION_MODEL).strip(),
        generation_fallback_models=split_models(
            os.getenv("GROQ_GENERATION_FALLBACK_MODELS"),
            DEFAULT_GENERATION_FALLBACK_MODELS,
        ),
        moderation_model=os.getenv("GROQ_MODERATION_MODEL", DEFAULT_MODERATION_MODEL).strip(),
        moderation_fallback_models=split_models(
            os.getenv("GROQ_MODERATION_FALLBACK_MODELS"),
            DEFAULT_MODERATION_FALLBACK_MODELS,
        ),
        timeout_seconds=float(os.getenv("GROQ_TIMEOUT_SECONDS", "30")),
    )


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()


def first_available_api_key() -> tuple[str | None, str | None]:
    for env_name in API_KEY_ENV_NAMES:
        value = os.getenv(env_name, "").strip()
        if value and not is_placeholder_api_key(value):
            return value, env_name
    return None, None


def is_placeholder_api_key(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return True
    return any(part in normalized for part in PLACEHOLDER_KEY_PARTS)


def split_models(raw_value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not raw_value:
        return default

    models = tuple(model.strip() for model in raw_value.split(",") if model.strip())
    return models or default


def unique_model_chain(primary_model: str, fallback_models: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered_models: list[str] = []
    for model in (primary_model, *fallback_models):
        if model and model not in seen:
            seen.add(model)
            ordered_models.append(model)
    return tuple(ordered_models)
