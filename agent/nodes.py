from __future__ import annotations

from typing import Any

from mcp_server.server import call_tool
from policy.engine import evaluate_query


def policy_node(query: str, model_name: str) -> dict[str, Any]:
    result = evaluate_query(query, model_name=model_name)
    return {
        "status": result.decision.lower(),  # allow | review | block
        "reason": result.reason,
        "risk_level": result.risk_level,
    }


def retrieval_node(query: str, k: int = 5) -> list[dict[str, Any]]:
    return call_tool("search_hybrid", query=query, k=k)


def confidence_node(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "confidence": "low",
            "top_score": 0.0,
            "avg_top3_score": 0.0,
        }

    top_score = float(results[0]["score"])
    top3 = results[:3]
    avg_top3_score = sum(float(r["score"]) for r in top3) / len(top3)

    if top_score >= 0.75 and avg_top3_score >= 0.55:
        confidence = "high"
    elif top_score >= 0.45:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "confidence": confidence,
        "top_score": top_score,
        "avg_top3_score": avg_top3_score,
    }


def summarize_evidence_node(results: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = []
    for r in results[:3]:
        evidence.append(
            {
                "source": r["source"],
                "score": r["score"],
                "preview": r["text"][:180],
            }
        )
    return {"evidence_summary": evidence}


def answer_mode_node(confidence: str, policy_status: str) -> dict[str, Any]:
    if confidence == "low":
        return {
            "answer_mode": "abstain",
            "answer_instruction": "Insufficient evidence in retrieved context; do not guess.",
        }

    if policy_status == "review":
        return {
            "answer_mode": "cautious",
            "answer_instruction": "Answer cautiously, use only evidence, and mention that the query was flagged for review.",
        }

    if confidence == "medium":
        return {
            "answer_mode": "cautious",
            "answer_instruction": "Answer cautiously, use only evidence, and explicitly mention uncertainty where needed.",
        }

    return {
        "answer_mode": "normal",
        "answer_instruction": "Answer normally using only the retrieved evidence.",
    }
