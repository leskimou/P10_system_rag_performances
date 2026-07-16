# utils/config.py
import os
from dotenv import load_dotenv

# Charger les variables d'environnement du fichier .env
load_dotenv()

# --- Clé API ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("⚠️ Attention: La clé API Mistral (MISTRAL_API_KEY) n'est pas définie dans le fichier .env")
    # Vous pouvez choisir de lever une exception ici ou de continuer avec des fonctionnalités limitées
    # raise ValueError("Clé API Mistral manquante. Veuillez la définir dans le fichier .env")

# --- Modèles Mistral ---
EMBEDDING_MODEL = "mistral-embed"
MODEL_NAME = "mistral-small-latest" # Ou un autre modèle comme mistral-large-latest

# --- Hyperparamètres du LLM (chatbot) ---
LLM_TEMPERATURE = 0.1      # Créativité des réponses (0.0 = déterministe, 1.0 = créatif)
LLM_TOP_P = 0.8            # Nucleus sampling : top-k n'est pas supporté par Mistral, top_p est l'équivalent
LLM_MAX_TOKENS = 2000      # Nombre maximum de tokens générés par réponse

# --- Configuration de l'Indexation ---
INPUT_DIR = "inputs"                # Dossier pour les données sources après extraction
VECTOR_DB_DIR = "vector_db"         # Dossier pour stocker l'index Faiss et les chunks
FAISS_INDEX_FILE = os.path.join(VECTOR_DB_DIR, "faiss_index.idx")
DOCUMENT_CHUNKS_FILE = os.path.join(VECTOR_DB_DIR, "document_chunks.pkl")

CHUNK_SIZE = 1500                # Taille des chunks en *caractères* (vise ~512 tokens)
CHUNK_OVERLAP = 150                 # Chevauchement en *caractères*
EMBEDDING_BATCH_SIZE = 32           # Taille des lots pour l'API d'embedding

# --- Configuration de la Recherche ---
SEARCH_K = 3                     # Nombre de documents à récupérer par défaut

# --- Configuration PostgreSQL (données NBA importées depuis Excel) ---
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PWD = os.getenv("DB_PWD")

if all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PWD]):
    POSTGRES_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PWD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    POSTGRES_URL = None

# --- Configuration de l'Application ---
APP_TITLE = "NBA Analyst AI"
NAME = "NBA" # Nom à personnaliser dans l'interface

# --- Observabilité ---
from .observability import configure_logfire

configure_logfire()