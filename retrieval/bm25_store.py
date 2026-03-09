from __future__ import annotations

from dataclasses import dataclass
from typing import List

from rank_bm25 import BM25Okapi
from retrieval.chunking import Chunk


@dataclass
class BM25Result:
    chunk: Chunk
    score: float


class BM25Store:
    def __init__(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        self.tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[BM25Result]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return [BM25Result(chunk=c, score=float(s)) for c, s in ranked]
