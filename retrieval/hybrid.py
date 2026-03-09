from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from retrieval.bm25_store import BM25Store
from retrieval.chunking import Chunk
from retrieval.faiss_store import FAISSStore


@dataclass
class HybridResult:
    chunk: Chunk
    score: float
    bm25_score: float
    faiss_score: float


class HybridRetriever:
    def __init__(
        self, chunks: List[Chunk], faiss_store: FAISSStore | None = None
    ) -> None:
        self.chunks = chunks
        self.bm25 = BM25Store(chunks)
        self.faiss = faiss_store or FAISSStore(chunks)
        if self.faiss.index is None:
            self.faiss.build()

    def search(self, query: str, top_k: int = 5) -> List[HybridResult]:
        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        faiss_results = self.faiss.search(query, top_k=top_k * 2)

        bm25_map: Dict[str, float] = {r.chunk.chunk_id: r.score for r in bm25_results}
        faiss_map: Dict[str, float] = {r.chunk.chunk_id: r.score for r in faiss_results}

        chunk_map: Dict[str, Chunk] = {}

        for r in bm25_results:
            chunk_map[r.chunk.chunk_id] = r.chunk
        for r in faiss_results:
            chunk_map[r.chunk.chunk_id] = r.chunk

        bm25_max = max(bm25_map.values(), default=1.0)
        faiss_max = max(faiss_map.values(), default=1.0)

        merged: List[HybridResult] = []
        for chunk_id, chunk in chunk_map.items():
            b = bm25_map.get(chunk_id, 0.0) / bm25_max if bm25_max else 0.0
            f = faiss_map.get(chunk_id, 0.0) / faiss_max if faiss_max else 0.0
            score = 0.5 * b + 0.5 * f

            merged.append(
                HybridResult(
                    chunk=chunk,
                    score=score,
                    bm25_score=b,
                    faiss_score=f,
                )
            )

        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[:top_k]
