from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_settings import load_llm_settings

TEST_PROMPT = "Reponds exactement avec le mot OK."


def main() -> int:
    settings = load_llm_settings()

    if not settings.groq_api_key:
        print("[ERREUR] Cle Groq absente ou placeholder dans .env.")
        print("Action: ouvre .env et renseigne GROQ_API_KEY avec ta vraie cle Groq.")
        print("Exemple: GROQ_API_KEY=gsk_...")
        return 2

    try:
        from groq import Groq
    except ImportError:
        print("[ERREUR] Le package groq n'est pas installe dans l'environnement Python.")
        print("Action: pip install -r requirements.txt")
        return 3

    client = Groq(api_key=settings.groq_api_key, timeout=settings.timeout_seconds)
    print(f"[INFO] Cle chargee depuis {settings.api_key_source}.")

    last_error: Exception | None = None
    for model in settings.generation_model_chain:
        print(f"[TEST] Verification du modele Groq: {model}")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": TEST_PROMPT}],
                temperature=0,
                max_completion_tokens=8,
            )
            content = (response.choices[0].message.content or "").strip()
            print(f"[OK] Cle Groq valide. Modele actif: {model}. Reponse: {content}")
            return 0
        except Exception as exc:  # Keep broad: SDK exceptions vary by version.
            last_error = exc
            print(f"[WARN] Echec avec {model}: {exc}")
            print("[INFO] Passage au modele fallback suivant si disponible.")

    print("[ERREUR] Aucun modele Groq configure n'a fonctionne.")
    print("Causes possibles: cle invalide, modele non autorise, quota/rate limit, reseau, ou nom de modele obsolete.")
    print("Action: verifie GROQ_API_KEY, puis essaie un autre modele dans GROQ_GENERATION_FALLBACK_MODELS.")
    if last_error is not None:
        print(f"Derniere erreur: {last_error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
