# Configuration API Groq et fallbacks LLM

## Fichier local `.env`

Le fichier `.env` est cree localement pour stocker la cle API Groq. Il est ignore par Git via `.gitignore` et ne doit jamais etre pousse sur GitHub.

A remplir :

```text
GROQ_API_KEY=gsk_...
```

Optionnel :

```text
GROQ_API_KEY_BACKUP=gsk_...
GROQ_API_KEY_FALLBACK=gsk_...
```

Le code lit les cles dans cet ordre :

1. `GROQ_API_KEY`
2. `GROQ_API_KEY_BACKUP`
3. `GROQ_API_KEY_FALLBACK`

## Choix des modeles

### Generation RAG

Modele principal :

```text
llama-3.3-70b-versatile
```

Raison : c'est le modele demande dans l'enonce du TP, et il fait partie des modeles de production Groq.

Fallbacks :

```text
openai/gpt-oss-120b,llama-3.1-8b-instant
```

- `openai/gpt-oss-120b` : fallback de qualite pour generation et raisonnement.
- `llama-3.1-8b-instant` : fallback rapide et economique pour verifier que l'application fonctionne.

### Moderation / prompt injection

Modele principal :

```text
openai/gpt-oss-safeguard-20b
```

Raison : modele de securite Groq adapte aux policies personnalisees de moderation et de prompt injection.

Fallbacks :

```text
meta-llama/llama-prompt-guard-2-86m,meta-llama/llama-prompt-guard-2-22m
```

Ces modeles sont dedies a la detection de prompt injection. Ils sont legers, mais peuvent etre moins flexibles qu'un modele policy-following.

## Tester la cle Groq

Depuis la racine du projet :

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\check_groq_api.py
```

Resultats possibles :

- `[OK] Cle Groq valide` : la cle fonctionne et au moins un modele repond.
- `[ERREUR] Cle Groq absente ou placeholder` : remplir `GROQ_API_KEY` dans `.env`.
- `[ERREUR] Le package groq n'est pas installe` : lancer `pip install -r requirements.txt`.
- `[ERREUR] Aucun modele Groq configure n'a fonctionne` : verifier la cle, le quota, le reseau ou changer les fallbacks.

## Regle de securite

Ne jamais commiter :

- `.env`
- une vraie cle API dans le code
- une vraie cle API dans le README ou les notebooks
