"""
Embedding Service – wraps the Sentence‑Transformer model for queries.
"""

from sentence_transformers import SentenceTransformer
from app.config import settings

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def encode(self, text: str):
        """Return embedding for a single text as a list of floats."""
        return self.model.encode(text).tolist()

    def encode_batch(self, texts: list[str]):
        """Return embeddings for a list of texts."""
        return self.model.encode(texts).tolist()