import chromadb
from chromadb.config import Settings
from app.config import settings
from app.services.embedding_service import EmbeddingService

class VectorStoreService:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_collection("support_tickets")
        self.embedder = EmbeddingService()

    def query(self, text: str, n_results: int = None):
        n = n_results or settings.TOP_K_RESULTS
        embedding = self.embedder.encode(text)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"]
        )
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }