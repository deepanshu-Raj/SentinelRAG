from __future__ import annotations

import json
import time
from pathlib import Path

from agent.graph import run_agent
from evaluation.metrics import (
    contains_forbidden_content,
    recall_at_k,
    reciprocal_rank,
    source_hit_rate,
)
from llm.answer import generate_answer
from retrieval.load_indices import load_hybrid_retriever


QUERIES_PATH = Path("evaluation/queries.json")

MODELS = [
    "gpt-5.4-2026-03-05",
    "gpt-5.1-2025-11-13",
    "gpt-4o-2024-11-20",
]


def load_queries() -> list[dict]:
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_naive_llm(query: str, model_name: str) -> dict:
    start = time.perf_counter()
    answer = generate_answer(query, [], model_name=model_name)
    latency = time.perf_counter() - start
    return {
        "answer": answer,
        "sources": [],
        "latency": latency,
        "blocked": False,
        "tool_calls": 0,
    }


def run_plain_rag(query: str, retriever, model_name: str) -> dict:
    start = time.perf_counter()
    results = retriever.search(query, top_k=5)
    contexts = [f"Source: {r.chunk.source}\n{r.chunk.text}" for r in results]
    answer = generate_answer(query, contexts, model_name=model_name)
    latency = time.perf_counter() - start
    return {
        "answer": answer,
        "sources": [r.chunk.source for r in results],
        "latency": latency,
        "blocked": False,
        "tool_calls": 1,
    }


def run_sentinelrag(query: str, model_name: str) -> dict:
    start = time.perf_counter()
    agent_result = run_agent(query, model_name=model_name)

    if agent_result["blocked"]:
        latency = time.perf_counter() - start
        return {
            "answer": "",
            "sources": [],
            "latency": latency,
            "blocked": True,
            "tool_calls": len(agent_result.get("tool_calls", [])),
            "reason": agent_result.get("reason", "Blocked by policy"),
        }

    if agent_result["answer_mode"] == "abstain":
        latency = time.perf_counter() - start
        return {
            "answer": "Insufficient evidence in retrieved context to answer confidently.",
            "sources": agent_result["sources"],
            "latency": latency,
            "blocked": False,
            "tool_calls": len(agent_result.get("tool_calls", [])),
        }

    contexts = [agent_result["context"]]
    answer = generate_answer(query, contexts, model_name=model_name)

    if agent_result.get("warning"):
        answer = f"[REVIEW FLAG] {agent_result['warning']}\n\n{answer}"

    latency = time.perf_counter() - start
    return {
        "answer": answer,
        "sources": agent_result["sources"],
        "latency": latency,
        "blocked": False,
        "tool_calls": len(agent_result.get("tool_calls", [])),
    }


def evaluate_mode(
    mode_name: str, runner, queries: list[dict], model_name: str, retriever=None
) -> list[dict]:
    rows = []

    for q in queries:
        query = q["query"]
        expected_sources = q["expected_sources"]
        qtype = q["type"]

        if retriever is not None:
            result = runner(query, retriever, model_name)
        else:
            result = runner(query, model_name)

        retrieved_sources = result["sources"]

        row = {
            "model": model_name,
            "mode": mode_name,
            "id": q["id"],
            "type": qtype,
            "query": query,
            "blocked": result["blocked"],
            "latency": result["latency"],
            "tool_calls": result["tool_calls"],
            "recall_at_5": recall_at_k(retrieved_sources, expected_sources, k=5),
            "mrr": reciprocal_rank(retrieved_sources, expected_sources),
            "source_hit_rate": source_hit_rate(retrieved_sources, expected_sources),
            "leakage": contains_forbidden_content(result["answer"]),
        }

        if qtype == "adversarial":
            row["policy_success"] = 1.0 if result["blocked"] else 0.0
        else:
            row["policy_success"] = 0.0

        rows.append(row)

    return rows


def main() -> None:
    queries = load_queries()
    retriever = load_hybrid_retriever()

    all_rows = []

    for model_name in MODELS:
        print(f"Running benchmark for {model_name}...")

        all_rows.extend(
            evaluate_mode(
                "naive_llm",
                run_naive_llm,
                queries,
                model_name=model_name,
            )
        )
        all_rows.extend(
            evaluate_mode(
                "plain_rag",
                run_plain_rag,
                queries,
                model_name=model_name,
                retriever=retriever,
            )
        )
        all_rows.extend(
            evaluate_mode(
                "sentinelrag",
                run_sentinelrag,
                queries,
                model_name=model_name,
            )
        )

    output_path = Path("evaluation/results_multimodel.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
