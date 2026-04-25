#!/usr/bin/env python
"""
Build a chunked vector store from outbound (company) replies.

- Reads raw twcs.csv
- Keeps only inbound=False (company replies)
- Drops missing text and very short replies (< 20 chars)
- Optionally samples a fixed number of replies (500k)
- Splits longer replies into chunks using sentence boundaries with overlap
- Generates embeddings for each chunk
- Stores chunks in ChromaDB (persistent)
"""

import sys
from pathlib import Path
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import nltk

nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

sys.path.insert(0, str(Path(__file__).parent.parent))

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
RAW_DATA_PATH = Path("data/raw/twcs.csv")
CHROMA_PERSIST_DIR = Path("chroma_data")
COLLECTION_NAME = "support_tickets"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 5000
MIN_REPLY_LENGTH = 20
MAX_CHUNK_SENTENCES = 5
OVERLAP_SENTENCES = 1
MAX_SOURCE_REPLIES = 500_000   # None = use all

# ------------------------------------------------------------------
# Helper: chunk text by sentences
# ------------------------------------------------------------------
def chunk_by_sentences(text: str, max_sentences: int, overlap: int):
    sentences = sent_tokenize(text)
    if not sentences:
        return []
    chunks = []
    start = 0
    while start < len(sentences):
        end = min(start + max_sentences, len(sentences))
        chunk_text = " ".join(sentences[start:end])
        chunks.append(chunk_text)
        start += (max_sentences - overlap)
        if start >= len(sentences):
            break
    return chunks

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("Loading raw data...")
    if not RAW_DATA_PATH.exists():
        print(f"❌ Raw data not found at {RAW_DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(RAW_DATA_PATH, usecols=['text', 'inbound'])
    print(f"Total rows: {len(df):,}")

    df = df[df['inbound'] == False].copy()
    print(f"Outbound rows: {len(df):,}")

    df = df.dropna(subset=['text'])
    df['text'] = df['text'].str.strip()
    df = df[df['text'].str.len() >= MIN_REPLY_LENGTH]
    print(f"After length filter (>= {MIN_REPLY_LENGTH} chars): {len(df):,}")

    # --- Sample a fixed number if set ---
    if MAX_SOURCE_REPLIES and len(df) > MAX_SOURCE_REPLIES:
        df = df.sample(n=MAX_SOURCE_REPLIES, random_state=42)
        print(f"Sampled down to {MAX_SOURCE_REPLIES:,} replies")

    print("Chunking replies...")
    all_chunks = []
    chunk_metadata = []  # <-- This was missing, causing the NameError

    for idx, row in df.iterrows():
        text = row['text']
        chunks = chunk_by_sentences(text, MAX_CHUNK_SENTENCES, OVERLAP_SENTENCES)
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_metadata.append({"source_reply_idx": idx})

    print(f"Total chunks: {len(all_chunks):,}")

    # Initialize embedding model
    print(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Connect to ChromaDB
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_PERSIST_DIR),
        settings=Settings(anonymized_telemetry=False)
    )

    # Delete existing collection
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Generate embeddings and add to ChromaDB in batches
    print("Generating embeddings and adding to ChromaDB...")
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]

    for start in range(0, len(all_chunks), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(all_chunks))
        batch_texts = all_chunks[start:end]
        batch_ids = ids[start:end]
        batch_meta = chunk_metadata[start:end]  # now defined

        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
        collection.add(
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
            ids=batch_ids
        )
        print(f"  Processed {end}/{len(all_chunks)} chunks")

    print(f"✅ Vector store built with {collection.count()} chunks.")
    print(f"   Data persisted to {CHROMA_PERSIST_DIR.absolute()}")

if __name__ == "__main__":
    main()