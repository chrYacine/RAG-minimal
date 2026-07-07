# RAG minimal

Mini-TP RAG realise en binome avec ChromaDB, sentence-transformers, Groq et un agent moderateur.

## Structure du projet

```text
RAG-minimal/
├── src/
│   ├── config.py
│   ├── vector_db.py
│   ├── moderator.py
│   └── rag.py
├── prompts/
│   ├── rag_system_prompt.txt
│   └── moderator_system_prompt.txt
├── data/
│   ├── corpus.py
│   └── 05_corpus_rag.csv
├── tests/
│   ├── test_vector_db.py
│   ├── test_moderator.py
│   └── test_rag.py
├── docs/
│   ├── 05_mini_TP_demo_mon_premier_RAG.pdf
│   └── ROADMAP_YACINE.md
├── .env.example
├── .gitignore
└── requirements.txt
```

## Installation

```powershell
cd C:\Users\yacch\Documents\first_rag
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Ajouter ensuite la cle Groq dans `.env`.

## Repartition

### Yacine - Partie 2

Branche de base : `dev/yacine-partie-2`

Responsabilites :

- `src/vector_db.py`
- `src/moderator.py`
- `prompts/moderator_system_prompt.txt`
- ChromaDB persistante
- embeddings sentence-transformers
- moderation JSON

### Adrien - Partie 1

Branche de base : `dev/adrien-partie-1`

Responsabilites :

- `src/rag.py`
- `prompts/rag_system_prompt.txt`
- appel Groq principal
- pipeline `answer_question()`
- tests finaux RAG

## Workflow Git

```powershell
git switch dev/yacine-partie-2
git pull
git switch -c feature/yacine-vector-db
```

Commit propre :

```powershell
git status
git add src/vector_db.py prompts/moderator_system_prompt.txt
git commit -m "feature: add vector database retrieval"
git push --set-upstream origin feature/yacine-vector-db
```

Conventions de branches :

- `feature/...` pour une fonctionnalite
- `test/...` pour les tests
- `documentation/...` pour la documentation
- `bug-fix/...` pour une correction
- `refactor/...` pour une amelioration interne sans changement de comportement

Eviter `git add *`. Ajouter seulement les fichiers concernes.

## Interfaces entre les deux parties

Yacine expose :

```python
chunks = vector_db.retrieve(question, n=3)
moderation = moderator.moderate(question)
```

Adrien utilise ces deux resultats pour construire le prompt RAG et bloquer les prompt injections avant l'appel au LLM principal.