from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval.chunking import Chunk


@dataclass
class FAISSResult:
    chunk: Chunk
    score: float


class FAISSStore:
    def __init__(
        self, chunks: List[Chunk], model_name: str = "all-MiniLM-L6-v2"
    ) -> None:
        self.chunks = chunks
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None

    def build(self) -> None:
        texts = [chunk.text for chunk in self.chunks]
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        embeddings = embeddings.astype("float32")
        dim = embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.index = index

    def save(self, path: str) -> None:
        if self.index is None:
            raise ValueError("Index has not been built")
        faiss.write_index(self.index, path)

    def load(self, path: str) -> None:
        self.index = faiss.read_index(path)

    def search(self, query: str, top_k: int = 5) -> List[FAISSResult]:
        if self.index is None:
            raise ValueError("Index is not built or loaded")

        q = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(q, top_k)

        results: List[FAISSResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(FAISSResult(chunk=self.chunks[idx], score=float(score)))
        return results
