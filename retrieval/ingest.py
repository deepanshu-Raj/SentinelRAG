from retrieval.chunking import load_and_chunk_directory
from retrieval.hybrid import HybridRetriever


def main() -> None:
    chunks = load_and_chunk_directory("data/raw")
    print(f"Loaded {len(chunks)} chunks")

    retriever = HybridRetriever(chunks)
    print("Built BM25 + FAISS hybrid retriever")

    sample_query = "How does FastAPI dependency injection work?"
    results = retriever.search(sample_query, top_k=3)

    print("\nSample retrieval results:\n")
    for i, result in enumerate(results, start=1):
        print(f"[{i}] score={result.score:.4f}")
        print(f"source={result.chunk.source}")
        print(result.chunk.text[:300])
        print("-" * 80)


if __name__ == "__main__":
    main()
