# RAG minimal

Mini-TP RAG realise en binome avec ChromaDB, sentence-transformers, Groq et un agent moderateur.

## Structure du projet

```text
RAG-minimal/
|-- src/
|   |-- config.py
|   |-- llm_settings.py
|   |-- vector_db.py
|   |-- moderator.py
|   `-- rag.py
|-- prompts/
|   |-- rag_system_prompt.txt
|   `-- moderator_system_prompt.txt
|-- data/
|   |-- corpus.py
|   `-- 05_corpus_rag.csv
|-- tests/
|   |-- test_vector_db.py
|   |-- test_moderator.py
|   `-- test_rag.py
|-- scripts/
|   `-- check_groq_api.py
|-- docs/
|   |-- API_AND_LLM_FALLBACKS.md
|   |-- ROADMAP_YACINE.md
|   `-- 05_mini_TP_demo_mon_premier_RAG.pdf
|-- .env.example
|-- .gitignore
`-- requirements.txt
```

## Installation

```powershell
cd C:\Users\yacch\Documents\first_rag
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Ajouter ensuite la cle Groq dans `.env` :

```text
GROQ_API_KEY=gsk_...
```

Le fichier `.env` est ignore par Git et ne doit jamais etre commite.

## Repartition

### Yacine - Partie 2

Branche de base : `dev/yacine-partie-2`

Responsabilites :

- `src/vector_db.py`
- `src/moderator.py`
- `prompts/moderator_system_prompt.txt`
- ChromaDB persistante
- embeddings sentence-transformers
- agent moderateur Groq
- configuration API et fallbacks LLM

### Adrien - Partie 1

Branche de base : `dev/adrien-partie-1`

Responsabilites :

- `src/rag.py`
- `prompts/rag_system_prompt.txt`
- appel Groq principal
- pipeline `answer_question()`
- tests comportementaux RAG

## Interfaces entre les deux parties

Yacine expose :

```python
chunks = vector_db.retrieve(question, n=3)
moderation = moderator.moderate(question)
```

Adrien utilise ces resultats pour construire le prompt RAG et bloquer les prompt injections avant l'appel au LLM principal.

## Verification API Groq

```powershell
python scripts\check_groq_api.py
```

La documentation des modeles, cles et fallbacks est dans `docs/API_AND_LLM_FALLBACKS.md`.

## Workflow Git

```powershell
git switch dev/yacine-partie-2
git pull
git switch -c feature/nom-clair
```

Commit propre :

```powershell
git status
git add fichier_precis.py autre_fichier.txt
git commit -m "feature: description claire"
git push --set-upstream origin feature/nom-clair
```

Conventions de branches :

- `feature/...` pour une fonctionnalite
- `test/...` pour les tests
- `documentation/...` pour la documentation
- `bug-fix/...` pour une correction
- `refactor/...` pour une amelioration interne sans changement de comportement

Eviter `git add *`. Ajouter seulement les fichiers concernes.
