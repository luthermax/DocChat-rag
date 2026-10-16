"""Appel du modèle de langage : Ollama (local) ou Groq (API gratuite avec quotas)."""
import os
import requests
import config

GROQ_URL = "https://api.groq.com/openai/v1"


class LLMError(Exception):
    """Erreur lisible, affichée telle quelle à l'utilisateur."""


def chat(messages, backend=None, model=None, api_key=None):
    backend = backend or config.LLM_BACKEND
    if backend == "groq":
        return _chat_groq(messages, model or config.GROQ_MODEL, api_key or os.getenv("GROQ_API_KEY"))
    return _chat_ollama(messages, model or config.OLLAMA_MODEL)


def _chat_ollama(messages, model):
    try:
        import ollama
        r = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0, "num_predict": config.MAX_ANSWER_TOKENS,
                     "num_ctx": config.OLLAMA_NUM_CTX},
        )
        return r["message"]["content"]
    except Exception as e:
        raise LLMError(f"Ollama ne répond pas, ou le modèle '{model}' est absent. "
                       f"Lance Ollama et fais : ollama pull {model}  (détail : {e})") from e


def _chat_groq(messages, model, api_key):
    if not api_key:
        raise LLMError("Clé API Groq manquante : colle-la dans la barre latérale.")
    try:
        r = requests.post(
            f"{GROQ_URL}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": messages, "temperature": 0,
                  "max_tokens": config.MAX_ANSWER_TOKENS},
            timeout=60,
        )
    except requests.RequestException as e:
        raise LLMError(f"Connexion à Groq impossible : {e}") from e
    if r.status_code == 401:
        raise LLMError("Clé API Groq refusée (erreur 401).")
    if r.status_code == 429:
        raise LLMError("Limite de requêtes Groq atteinte (erreur 429) : attends un peu, puis réessaie.")
    if r.status_code != 200:
        raise LLMError(f"Erreur Groq {r.status_code} : {r.text[:200]}")
    return r.json()["choices"][0]["message"]["content"]


def list_groq_models(api_key):
    """Modèles de texte disponibles pour cette clé (la liste change avec le temps)."""
    try:
        r = requests.get(f"{GROQ_URL}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
    except requests.RequestException as e:
        raise LLMError(f"Connexion à Groq impossible : {e}") from e
    if r.status_code != 200:
        raise LLMError(f"Impossible de lister les modèles Groq (erreur {r.status_code}).")
    ids = [m["id"] for m in r.json().get("data", [])]
    return sorted(i for i in ids if not any(x in i.lower() for x in ("whisper", "tts", "guard", "embed")))


def ollama_ready():
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False
