from __future__ import annotations

import sys

from llm.answer import generate_answer
from retrieval.load_indices import load_hybrid_retriever
from retrieval.hybrid import HybridRetriever


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m retrieval.test_query "your query"')

    query = sys.argv[1]
    retriever = load_hybrid_retriever()
    results = retriever.search(query, top_k=5)

    print(f"\nQuery: {query}\n")
    contexts: list[str] = []

    for i, result in enumerate(results, start=1):
        print(
            f"[{i}] hybrid={result.score:.4f} bm25={result.bm25_score:.4f} faiss={result.faiss_score:.4f}"
        )
        print(f"source: {result.chunk.source}")
        print(result.chunk.text[:400])
        print("=" * 100)
        contexts.append(f"Source: {result.chunk.source}\n{result.chunk.text}")

    print("\nGenerating answer...\n")
    answer = generate_answer(query, contexts)
    print(answer)
    print()


if __name__ == "__main__":
    main()
