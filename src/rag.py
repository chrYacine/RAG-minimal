from pathlib import Path
from typing import Any, Callable


DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "rag_system_prompt.txt"
DEFAULT_TOP_K = 3
FALLBACK_PROMPT_TEMPLATE = """Tu es un assistant RAG.
Reponds uniquement avec les informations du contexte.
Si la reponse n'est pas dans le contexte, reponds que tu ne sais pas d'apres le corpus fourni.

Contexte :
{{Chunks}}

Question :
{{question}}
"""


class RAG:
    """Pipeline RAG: moderation, retrieval, prompt building, then generation."""

    def __init__(
        self,
        vector_db: Any,
        moderator: Any | None = None,
        generator: Callable[[str], str] | None = None,
        prompt_path: str | Path = DEFAULT_PROMPT_PATH,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.vector_db = vector_db
        self.moderator = moderator
        self.generator = generator
        self.prompt_path = Path(prompt_path)
        self.top_k = top_k

    def answer_question(self, question: str) -> str:
        """Return an answer based only on retrieved chunks."""
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("La question ne peut pas etre vide.")

        moderation = self._moderate(cleaned_question)
        if self._is_blocked(moderation):
            return "Je ne peux pas traiter cette demande car elle ne respecte pas les regles du RAG."

        chunks = self._retrieve_chunks(cleaned_question)
        prompt = self.build_prompt(chunks, cleaned_question)

        if self.generator is None:
            return self._generate_answer(prompt)
        return self.generator(prompt)

    def build_prompt(self, chunks: list[dict[str, Any]], question: str) -> str:
        """Build the final prompt sent to the language model."""
        template = self._load_prompt_template()
        formatted_chunks = self._format_chunks(chunks)
        return (
            template.replace("{{Chunks}}", formatted_chunks)
            .replace("{{chunks}}", formatted_chunks)
            .replace("{{question}}", question)
        )

    def _moderate(self, question: str) -> dict[str, Any]:
        if self.moderator is None:
            return {"allowed": True}
        result = self.moderator.moderate(question)
        if isinstance(result, dict):
            return result
        return {"allowed": bool(result)}

    def _is_blocked(self, moderation: dict[str, Any]) -> bool:
        blocked_keys = ("blocked", "is_blocked", "injection_detected")
        allowed_keys = ("allowed", "is_allowed", "safe", "is_safe")

        if any(bool(moderation.get(key)) for key in blocked_keys):
            return True
        if any(key in moderation and not bool(moderation.get(key)) for key in allowed_keys):
            return True
        return False

    def _retrieve_chunks(self, question: str) -> list[dict[str, Any]]:
        raw_chunks = self.vector_db.retrieve(question, n=self.top_k)
        return [self._normalize_chunk(chunk) for chunk in raw_chunks]

    def _normalize_chunk(self, chunk: Any) -> dict[str, Any]:
        if isinstance(chunk, dict):
            return chunk
        return {"text": str(chunk)}

    def _format_chunks(self, chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            return "Aucun chunk pertinent trouve."

        lines = []
        for index, chunk in enumerate(chunks, start=1):
            text = chunk.get("text") or chunk.get("document") or ""
            source = chunk.get("source")
            category = chunk.get("categorie") or chunk.get("category")

            metadata = []
            if source:
                metadata.append(f"source={source}")
            if category:
                metadata.append(f"categorie={category}")

            suffix = f" ({', '.join(metadata)})" if metadata else ""
            lines.append(f"Chunk {index}{suffix}: {text}")
        return "\n".join(lines)

    def _load_prompt_template(self) -> str:
        if not self.prompt_path.exists():
            return FALLBACK_PROMPT_TEMPLATE

        template = self.prompt_path.read_text(encoding="utf-8").strip()
        if not template:
            return FALLBACK_PROMPT_TEMPLATE
        return template

    def _generate_answer(self, prompt: str) -> str:
        raise NotImplementedError(
            "La generation Groq sera ajoutee dans la branche feature/groq-generation."
        )
