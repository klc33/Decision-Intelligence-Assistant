#!/usr/bin/env python
"""
Build Vector Store – Phase 3

This script:
1. Loads the cleaned labeled dataset (500k rows).
2. Generates embeddings for each tweet using a local SentenceTransformer model.
3. Stores embeddings and metadata in ChromaDB.
4. Persists the ChromaDB data to ./chroma_data/

Usage:
    python scripts/build_vector_store.py
"""

import sys
from pathlib import Path
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Configuration
# ============================================================================
DATA_PATH = Path("data/processed/tickets_with_labels.csv")
CHROMA_PERSIST_DIR = Path("chroma_data")
COLLECTION_NAME = "support_tickets"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dimensions, fast, local
BATCH_SIZE = 5000  # Process in batches to manage memory


def main():
    print("=" * 60)
    print("Phase 3: Building Vector Store")
    print("=" * 60)

    # 1. Load cleaned data
    print("\n📂 Loading cleaned data...")
    if not DATA_PATH.exists():
        print(f"❌ Data not found at {DATA_PATH}")
        print("   Please run Phase 1 first: python scripts/prepare_data.py")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"   Loaded {len(df):,} rows")

    # 2. Initialize embedding model
    print(f"\n🧠 Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # 3. Initialize ChromaDB (persistent)
    print(f"\n💾 Initializing ChromaDB at {CHROMA_PERSIST_DIR.absolute()}")
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(CHROMA_PERSIST_DIR),
        settings=Settings(anonymized_telemetry=False)
    )

    # 4. Create or reset collection
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"   Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"   Created collection '{COLLECTION_NAME}'")

    # 5. Generate embeddings and add in batches
    print(f"\n🔨 Generating embeddings and adding to ChromaDB...")
    texts = df['text'].tolist()
    priorities = df['priority'].tolist()
    ids = [f"ticket_{i}" for i in range(len(df))]

    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx in range(0, len(texts), BATCH_SIZE):
        start = batch_idx
        end = min(batch_idx + BATCH_SIZE, len(texts))
        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        batch_priorities = priorities[start:end]

        # Generate embeddings
        embeddings = model.encode(
            batch_texts,
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()

        # Add to collection
        collection.add(
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=[{"priority": p} for p in batch_priorities],
            ids=batch_ids
        )

        # Progress update
        batch_num = batch_idx // BATCH_SIZE + 1
        print(f"   Processed batch {batch_num}/{total_batches} ({end:,} documents)")

    print(f"\n✅ Vector store built successfully!")
    print(f"   Collection '{COLLECTION_NAME}' contains {collection.count():,} documents")
    print(f"   Data persisted to {CHROMA_PERSIST_DIR.absolute()}")


if __name__ == "__main__":
    main()