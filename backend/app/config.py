import os
from pathlib import Path
from dotenv import load_dotenv

# Project root is 3 levels up from this file:
# backend/app/config.py → backend/app → backend → project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Load .env from project root
load_dotenv(PROJECT_ROOT / ".env")

class Settings:
    # Groq
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # ChromaDB – force absolute path
    _chroma_dir = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_data"))
    CHROMA_PERSIST_DIR: str = str(Path(_chroma_dir).resolve())  # absolute

    # Embedding
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # RAG
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "3"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

    # ML Model path
    MODEL_PATH: Path = Path(__file__).parent / "models" / "priority_classifier.pkl"

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: Path = PROJECT_ROOT / "backend" / "app" / "data" / "logs"

settings = Settings()