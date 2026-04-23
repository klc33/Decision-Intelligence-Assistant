import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root (two levels up from this file)
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

class Settings:
    # Groq
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    
    # ChromaDB
    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8001"))
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    
    # Embedding
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # RAG
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "3"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
    
    # Model path
    MODEL_PATH: Path = Path(__file__).parent / "models" / "priority_classifier.pkl"
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: Path = Path(__file__).parent / "data" / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()