from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "jsons"
EMBEDDINGS_PATH = ROOT_DIR / "embeddings.joblib"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
REQUEST_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))
DEFAULT_TOP_K = 5
MAX_TOP_K = 20
