# Mon premier RAG

Mini-TP RAG realise en binome avec ChromaDB, sentence-transformers, Groq et un agent moderateur.

## Structure du projet

```text
mon-premier-rag/
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
│   └── 05_mini_TP_demo_mon_premier_RAG.pdf
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Repartition

### Partie 1 - Binome

Responsabilites :

- `src/rag.py`
- `prompts/rag_system_prompt.txt`
- appel Groq
- pipeline `answer_question()`
- tests finaux RAG

Branche de base : `dev/binome-partie-1`

Branches conseillees :

- `feature/rag-pipeline`
- `feature/groq-generation`
- `test/rag-final-questions`

### Partie 2 - Toi

Responsabilites :

- `src/vector_db.py`
- `src/moderator.py`
- `prompts/moderator_system_prompt.txt`
- ChromaDB persistante
- embeddings sentence-transformers
- moderation JSON

Branche de base : `dev/moi-partie-2`

Branches conseillees :

- `feature/vector-db`
- `feature/moderator-agent`
- `test/vector-db-retrieval`

## Workflow Git

Avant une nouvelle fonctionnalite :

```powershell
git switch dev/moi-partie-2
git pull
git switch -c feature/vector-db
```

Commit propre :

```powershell
git status
git add src/vector_db.py tests/test_vector_db.py
git commit -m "feature: add vector database retrieval"
git push --set-upstream origin feature/vector-db
```

Conventions de branches :

- `feature/...` pour une fonctionnalite
- `test/...` pour les tests
- `documentation/...` pour la documentation
- `bug-fix/...` pour une correction
- `refactor/...` pour une amelioration interne sans changement de comportement

Eviter `git add *`. Ajouter seulement les fichiers concernes.

## GitHub

Creer un repository GitHub vide, puis lier ce depot local :

```powershell
git remote add origin https://github.com/<user>/mon-premier-rag.git
git push -u origin main
git push -u origin dev/moi-partie-2
git push -u origin dev/binome-partie-1
```

Chaque fonctionnalite doit passer par une Pull Request relue par l'autre membre du binome.