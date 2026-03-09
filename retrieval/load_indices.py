from __future__ import annotations

import json
from pathlib import Path

from retrieval.chunking import chunk_from_dict
from retrieval.faiss_store import FAISSStore
from retrieval.hybrid import HybridRetriever

PROCESSED_DIR = Path("data/processed")
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"
FAISS_FILE = PROCESSED_DIR / "faiss.index"


def load_hybrid_retriever() -> HybridRetriever:
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = [chunk_from_dict(x) for x in json.load(f)]

    faiss_store = FAISSStore(chunks)
    faiss_store.load(str(FAISS_FILE))

    return HybridRetriever(chunks=chunks, faiss_store=faiss_store)
