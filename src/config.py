import os

# --- Dossiers ---
RAW_DIR = "data/raw"          # PDF de démonstration et d'évaluation
CACHE_DIR = "index/docs"      # index sauvegardé de chaque document (un dossier par document)

# --- Limites (documents chargés depuis l'interface) ---
MAX_PAGES = 300               # pages maximum par document
MAX_DOCS = 5                  # documents chargés en même temps dans une session

# --- Découpage (ingest.py) ---
CHUNK_SIZE = 1000             # taille maximale d'un passage, en caractères
CHUNK_OVERLAP = 150           # chevauchement quand une section doit être coupée
MIN_CHUNK = 60                # les passages plus courts sont ignorés
X_TOLERANCE = 2               # seuil d'espacement de pdfplumber

# --- Embeddings ---
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# --- Recherche (retrieve.py) ---
TOP_K = 4                     # passages retrouvés par question
NEIGHBOR_TOP = 1              # les N meilleurs passages reçoivent leurs voisins (avant/après)
BM25_WEIGHT = 1.0             # poids de la recherche par mots exacts (0 = désactivée)
MIN_SIM = 0.0                 # sous ce score de similarité, on ne lance pas le LLM (0 = désactivé)

# --- Modèle de langage (llm.py) ---
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")   # "ollama" (local) ou "groq" (API)
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_NUM_CTX = 4096         # fenêtre de contexte demandée à Ollama
GROQ_MODEL = "llama-3.3-70b-versatile"   # à vérifier : la liste des modèles change
MAX_ANSWER_TOKENS = 800

# --- Contrôle des réponses (generate.py) ---
STRICT_GROUNDING = False      # True : une réponse non appuyée par les extraits est remplacée par un refus
