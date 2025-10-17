"""
Configuration centralisée pour le pipeline RAG avec FAISS + helpers Ollama
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict, Any

# ------------------------ Chemins de base ------------------------
BASE_DIR = Path(__file__).parent.parent
DOCUMENTS_PATH = BASE_DIR / "data" / "raw"
OUTPUT_PATH = BASE_DIR / "data" / "processed"
INDEX_PATH = BASE_DIR / "data" / "index"
RUNTIME_PATH = BASE_DIR / "data" / "runtime"

# ------------------------ Fichiers de données ------------------------
CHUNKS_JSON_PATH = OUTPUT_PATH / "chunks.json"
EMBEDDINGS_JSON_PATH = OUTPUT_PATH / "embeddings.json"

# ------------------------ Fichiers FAISS ------------------------
FAISS_INDEX_PATH = INDEX_PATH / "faiss.index"
IDMAP_PATH = INDEX_PATH / "idmap.json"
METADATA_PATH = INDEX_PATH / "metadata.json"

# ------------------------ Config chunking/tokens ------------------------
TARGET_TOKENS = 50
MAX_TOKENS = 60
MIN_TOKENS = 40
OVERLAP_TOKENS = 0          # pas de duplication
TOKENS_PER_WORD = 1.3

# ------------------------ Config embeddings ------------------------
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ------------------------ Config FAISS ------------------------
FAISS_INDEX_TYPE = "IndexFlatIP"  # Inner Product => cosine
FAISS_NPROBE = 10             # si IVF (inutile ici, mais gardé)
FAISS_NORMALIZE_VECTORS = True

# ------------------------ Config Retrieval/Generation ------------------------
RETRIEVAL_TOP_K = 20
RETRIEVAL_MIN_SCORE = 0.30
RETRIEVAL_NORMALIZE = True
RETRIEVAL_METRIC = "ip"     # inner product

MAX_RESPONSE_CHARS = 1800
RESPONSE_LANGUAGE = "fr"

SUPPORTED_EXTENSIONS = ['.txt', '.json', '.docx', '.pdf', '.doc']

# ------------------------ Détection de sections ------------------------
SECTION_MIN_LENGTH = 10
SECTION_MAX_TITLE_LENGTH = 80
HEADING_PATTERNS = [
    r'^#+\s+',               # Markdown headers
    r'^\d+\.\s+',            # Numbered headers
    r'^[A-Z0-9][\.\)][^\n]+$',
    r'^[A-Z][A-Za-z\s]{0,20}:$',
    r'^Environnement\s+[A-Z]+',
    r'description\s+technique',
    r'webMethods',
    r'Software\s+AG',
]

# ------------------------ Logs ------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class RAGSettings:
    """Singleton de configuration RAG"""

    def __init__(self):
        # Création des dossiers nécessaires
        DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        INDEX_PATH.mkdir(parents=True, exist_ok=True)
        RUNTIME_PATH.mkdir(parents=True, exist_ok=True)

    # --- Paths (exposés sous forme de propriétés) ---
    @property
    def documents_path(self) -> Path:
        return DOCUMENTS_PATH

    @property
    def output_path(self) -> Path:
        return OUTPUT_PATH

    @property
    def index_path(self) -> Path:
        return INDEX_PATH

    @property
    def runtime_path(self) -> Path:
        return RUNTIME_PATH

    @property
    def chunks_path(self) -> Path:
        return CHUNKS_JSON_PATH

    @property
    def embeddings_path(self) -> Path:
        return EMBEDDINGS_JSON_PATH

    @property
    def faiss_index_path(self) -> Path:
        return FAISS_INDEX_PATH

    @property
    def idmap_path(self) -> Path:
        return IDMAP_PATH

    @property
    def metadata_path(self) -> Path:
        return METADATA_PATH

    # --- Config retrieval/FAISS ---
    def get_retrieval_config(self) -> dict:
        return {
            "top_k": 6,
            "min_score": 0.30,
            "normalize": True,
            "w_sim": 0.6,
            "w_kw": 0.25,
            "w_title": 0.15,
            "min_cosine": 0.28,
            "min_final": 0.35,
            "llm_rerank": True,
            "llm_rerank_k": 10
        }

    def get_faiss_config(self) -> dict:
        return {
            "index_type": FAISS_INDEX_TYPE,
            "nprobe": FAISS_NPROBE,
            "normalize": FAISS_NORMALIZE_VECTORS,
            "dimension": EMBEDDING_DIMENSION,
        }


# Instance globale
settings = RAGSettings()


# ------------------------ Compatibilité (si nécessaire) ------------------------
class Config:
    """Compat couche legacy"""
    RETRIEVAL_MIN_SCORE = RETRIEVAL_MIN_SCORE
    RETRIEVAL_NORMALIZE = RETRIEVAL_NORMALIZE


# ======================================================================
#                           O L L A M A   H E L P E R S
# ======================================================================

# Essayez langchain-ollama en priorité; fallback vers community si absent
try:
    from langchain_ollama import ChatOllama          # pip install langchain-ollama
    OLLAMA_AVAILABLE = True
except Exception:
    try:
        from langchain_community.chat_models import ChatOllama  # older path
        OLLAMA_AVAILABLE = True
    except Exception:
        OLLAMA_AVAILABLE = False

# HTTP client pour check_ollama_status
try:
    import requests
    _REQUESTS_AVAILABLE = True
except Exception:
    _REQUESTS_AVAILABLE = False


class OllamaConfig:
    """Paramétrage par défaut pour deepseek-coder:instruct via Ollama"""
    DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-coder:instruct")
    DEFAULT_CONFIG = {
        "model": DEFAULT_MODEL,
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "6144")),
        "stop": ["```", "Human:", "User:", "\n\nHuman:", "\n\nUser:"],
        # "base_url": os.getenv("OLLAMA_HOST", "http://localhost:11434"),  # for some versions
    }

    def __init__(self, model: Optional[str] = None, **kwargs):
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Ollama non disponible. Installez: pip install langchain-ollama")
        self.model = model or self.DEFAULT_MODEL
        self.config = {**self.DEFAULT_CONFIG, **kwargs}

    def create_llm(self, **override) -> "ChatOllama":
        cfg = {**self.config, **override}
        llm = ChatOllama(**cfg)  # type: ignore
        # mini-ping pour valider la connexion
        try:
            _ = llm.invoke("ping")
        except Exception as e:
            raise RuntimeError(f"Echec ping Ollama: {e}")
        return llm


def create_ollama_llm(model: Optional[str] = None, **kwargs) -> "ChatOllama":
    """
    Crée une instance ChatOllama prête à l'emploi (avec ping).
    """
    return OllamaConfig(model, **kwargs).create_llm()


def get_deepseek_coder_config() -> Dict[str, Any]:
    """
    Configuration recommandée pour deepseek-coder:instruct (RAG/code).
    """
    return {
        "model": os.getenv("OLLAMA_MODEL", "deepseek-coder:instruct"),
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "6144")),
        "stop": ["```", "Human:", "User:", "\n\nHuman:", "\n\nUser:"],
        # "base_url": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    }


def check_ollama_status() -> Dict[str, Any]:
    """
    Vérifie si Ollama répond et liste les modèles disponibles.
    """
    status: Dict[str, Any] = {
        "ollama_running": False,
        "models_available": [],
        "deepseek_available": False,
        "error": None,
    }

    if not _REQUESTS_AVAILABLE:
        status["error"] = "Le module 'requests' n'est pas installé."
        return status

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        if r.status_code == 200:
            status["ollama_running"] = True
            data = r.json()
            models = [m.get("name") for m in data.get(
                "models", []) if isinstance(m, dict)]
            status["models_available"] = models
            status["deepseek_available"] = any(
                "deepseek-coder" in (m or "") for m in models)
        else:
            status["error"] = f"HTTP {r.status_code}"
    except Exception as e:
        status["error"] = str(e)

    return status
