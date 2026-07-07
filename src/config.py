from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROMPTS_DIR = ROOT_DIR / "prompts"
CHROMA_PERSIST_DIR = ROOT_DIR / "chroma_db"

CORPUS_CSV_PATH = DATA_DIR / "05_corpus_rag.csv"
MODERATOR_PROMPT_PATH = PROMPTS_DIR / "moderator_system_prompt.txt"

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_minimal_chunks")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "distiluse-base-multilingual-cased-v2",
)
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

MODERATION_MODEL_NAME = os.getenv("GROQ_MODERATION_MODEL", "openai/gpt-oss-safeguard-20b")
MODERATION_USE_JSON_MODE = os.getenv("GROQ_MODERATION_USE_JSON_MODE", "false").lower() in {
    "1",
    "true",
    "yes",
}
MODERATION_MAX_TOKENS = int(os.getenv("GROQ_MODERATION_MAX_TOKENS", "256"))

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
