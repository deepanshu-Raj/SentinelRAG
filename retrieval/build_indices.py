from __future__ import annotations

import json
from pathlib import Path

from retrieval.chunking import chunk_to_dict, load_and_chunk_directory
from retrieval.faiss_store import FAISSStore


PROCESSED_DIR = Path("data/processed")
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"
FAISS_FILE = PROCESSED_DIR / "faiss.index"


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_and_chunk_directory("data/raw")
    print(f"Loaded {len(chunks)} chunks")

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump([chunk_to_dict(c) for c in chunks], f, indent=4)

    faiss_store = FAISSStore(chunks)
    faiss_store.build()
    faiss_store.save(str(FAISS_FILE))

    print(f"Saved chunked metadata to {CHUNKS_FILE}")
    print(f"Saved FAISS index to {FAISS_FILE}")


if __name__ == "__main__":
    main()
