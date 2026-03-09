from __future__ import annotations

from typing import Any

from agent.nodes import (
    answer_mode_node,
    confidence_node,
    policy_node,
    retrieval_node,
    summarize_evidence_node,
)


def run_agent(query: str, model_name: str) -> dict[str, Any]:
    state: dict[str, Any] = {
        "query": query,
        "blocked": False,
        "warning": None,
        "reason": None,
        "risk_level": None,
        "sources": [],
        "context": "",
        "retrieved_chunks": [],
        "evidence_summary": [],
        "tool_calls": [],
        "confidence": None,
        "top_score": 0.0,
        "avg_top3_score": 0.0,
        "answer_mode": None,
        "answer_instruction": None,
    }

    policy = policy_node(query, model_name=model_name)
    state.update(policy)

    if state["status"] == "block":
        state["blocked"] = True
        return state

    if state["status"] == "review":
        state["warning"] = f"Flagged for review: {state['reason']}"

    results = retrieval_node(query, k=5)
    state["tool_calls"].append("search_hybrid")
    state["retrieved_chunks"] = results

    confidence = confidence_node(results)
    state.update(confidence)

    evidence = summarize_evidence_node(results)
    state.update(evidence)

    answer_mode = answer_mode_node(
        confidence=state["confidence"],
        policy_status=state["status"],
    )
    state.update(answer_mode)

    state["sources"] = [r["source"] for r in results]
    state["context"] = "\n\n".join(
        f"Source: {r['source']}\n{r['text']}" for r in results
    )

    return state
