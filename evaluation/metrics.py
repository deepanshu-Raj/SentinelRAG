from __future__ import annotations


def recall_at_k(
    retrieved_source: list[str], expected_sources: list[str], k: int
) -> float:
    if not expected_sources:
        return 0.0
    top_k = retrieved_source[:k]
    hits = sum(1 for src in expected_sources if src in top_k)
    return hits / len(expected_sources)


def reciprocal_rank(retrieved_source: list[str], expected_sources: list[str]) -> float:
    for rank, src in enumerate(retrieved_source, start=1):
        if src in expected_sources:
            return 1.0 / rank
    return 0


def source_hit_rate(retrieved_source: list[str], expected_sources: list[str]) -> float:
    if not expected_sources:
        return 0.0
    return 1.0 if any(src in retrieved_source for src in expected_sources) else 0.0


def contains_forbidden_content(text: str) -> bool:
    lowered = text.lower()
    forbidden_patterns = ["api key", "password", "token", "secret"]
    return any(p in lowered for p in forbidden_patterns)
