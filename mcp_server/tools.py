from retrieval.load_indices import load_hybrid_retriever

retriever = load_hybrid_retriever()


def search_hybrid(query: str, k: int = 5):
    results = retriever.search(query, top_k=k)

    return [
        {
            "source": r.chunk.source,
            "text": r.chunk.text,
            "score": r.score,
        }
        for r in results
    ]
