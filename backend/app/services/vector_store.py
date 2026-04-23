import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from backend.app.config import settings
from typing import List, Dict, Any

class VectorStoreService:
    def __init__(self):
        # Connect to persistent ChromaDB
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_collection("support_tickets")
        
        # Embedding model (same as used in build_vector_store)
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
    
    def query(self, text: str, n_results: int = None) -> Dict[str, Any]:
        n = n_results or settings.TOP_K_RESULTS
        embedding = self.model.encode(text).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"]
        )
        # Flatten lists (Chroma returns lists of lists)
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }