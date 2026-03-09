from __future__ import annotations

from pathlib import Path

import streamlit as st

import pandas as pd

from agent.graph import run_agent
from llm.answer import generate_answer
from retrieval.load_indices import load_hybrid_retriever


MODELS = [
    "gpt-5.4-2026-03-05",
    "gpt-5.1-2025-11-13",
    "gpt-4o-2024-11-20",
]

MODES = [
    "naive_llm",
    "plain_rag",
    "sentinelrag",
]


st.set_page_config(page_title="SentinelRAG", layout="wide")
st.title("SentinelRAG")
st.caption(
    "Policy-aware retrieval agent with hybrid RAG, MCP-style tool routing, and multi-model evaluation"
)

st.markdown(
    """
    SentinelRAG is a **policy-aware retrieval agent** that prevents sensitive information leakage in RAG systems.

    The benchmark below compares three systems:

    - **naive_llm** – direct LLM answering (no retrieval)
    - **plain_rag** – standard retrieval augmented generation
    - **sentinelrag** – policy-aware retrieval agent

    SentinelRAG reduces **data leakage** while maintaining **retrieval quality**.
    """
)


@st.cache_resource
def get_retriever():
    return load_hybrid_retriever()


retriever = get_retriever()


def run_naive_llm(query: str, model_name: str) -> dict:
    answer = generate_answer(query, [], model_name=model_name)
    return {
        "answer": answer,
        "sources": [],
        "policy_status": "N/A",
        "policy_reason": "No policy layer in naive mode",
        "tool_calls": [],
        "evidence_summary": [],
    }


def run_plain_rag(query: str, model_name: str) -> dict:
    results = retriever.search(query, top_k=5)

    contexts = [f"Source: {r.chunk.source}\n{r.chunk.text}" for r in results]
    answer = generate_answer(query, contexts, model_name=model_name)

    evidence_summary = [
        {
            "source": r.chunk.source,
            "score": round(r.score, 4),
            "preview": r.chunk.text[:180],
        }
        for r in results[:3]
    ]

    return {
        "answer": answer,
        "sources": [r.chunk.source for r in results],
        "policy_status": "N/A",
        "policy_reason": "No policy layer in plain RAG mode",
        "tool_calls": ["search_hybrid"],
        "evidence_summary": evidence_summary,
    }


def run_sentinelrag(query: str, model_name: str) -> dict:
    agent_result = run_agent(query, model_name=model_name)

    if agent_result["blocked"]:
        return {
            "answer": "Request blocked by policy.",
            "sources": [],
            "policy_status": agent_result.get("status", "block").upper(),
            "policy_reason": agent_result.get("reason", "Blocked by policy"),
            "tool_calls": agent_result.get("tool_calls", []),
            "evidence_summary": [],
        }

    if agent_result["answer_mode"] == "abstain":
        return {
            "answer": "Insufficient evidence in retrieved context to answer confidently.",
            "sources": agent_result["sources"],
            "policy_status": agent_result.get("status", "allow").upper(),
            "policy_reason": agent_result.get("reason", ""),
            "tool_calls": agent_result.get("tool_calls", []),
            "evidence_summary": agent_result.get("evidence_summary", []),
        }

    answer = generate_answer(query, [agent_result["context"]], model_name=model_name)

    if agent_result.get("warning"):
        answer = f"[REVIEW FLAG] {agent_result['warning']}\n\n{answer}"

    return {
        "answer": answer,
        "sources": agent_result["sources"],
        "policy_status": agent_result.get("status", "allow").upper(),
        "policy_reason": agent_result.get("reason", ""),
        "tool_calls": agent_result.get("tool_calls", []),
        "evidence_summary": agent_result.get("evidence_summary", []),
    }


st.sidebar.header("Configuration")
selected_model = st.sidebar.selectbox("Select model", MODELS, index=0)
selected_mode = st.sidebar.selectbox("Select system mode", MODES, index=2)

st.sidebar.markdown("---")
show_plots = st.sidebar.checkbox("Show benchmark plots", value=True)

query = st.text_area(
    "Ask a question about the FastAPI corpus",
    value="How does FastAPI dependency injection work?",
    height=100,
)

run_button = st.button("Run Query")


if run_button:
    with st.spinner("Running..."):
        if selected_mode == "naive_llm":
            result = run_naive_llm(query, selected_model)
        elif selected_mode == "plain_rag":
            result = run_plain_rag(query, selected_model)
        else:
            result = run_sentinelrag(query, selected_model)

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.subheader("Answer")
        st.write(result["answer"])

        if result["sources"]:
            st.subheader("Sources")
            for src in result["sources"]:
                st.code(src)

    with col2:
        st.subheader("System Details")
        st.markdown(f"**Mode:** {selected_mode}")
        st.markdown(f"**Model:** {selected_model}")
        st.markdown(f"**Policy Status:** {result['policy_status']}")
        st.markdown(f"**Policy Reason:** {result['policy_reason']}")

        st.markdown("**Tool Calls:**")
        if result["tool_calls"]:
            for t in result["tool_calls"]:
                st.write(f"- {t}")
        else:
            st.write("None")

    if result["evidence_summary"]:
        st.subheader("Evidence Summary")
        for idx, item in enumerate(result["evidence_summary"], start=1):
            with st.expander(
                f"{idx}. {item['source']} | score={item.get('score', 'N/A')}"
            ):
                st.write(item["preview"])

st.markdown("---")

if show_plots:
    st.header("Benchmark Results")

    plot_dir = Path("artifacts")
    retrieval_plot = plot_dir / "retrieval_metrics_comparison.png"
    safety_plot = plot_dir / "safety_latency_comparison.png"
    summary_csv = plot_dir / "summary_metrics.csv"

    col1, col2 = st.columns(2)

    with col1:
        if retrieval_plot.exists():
            st.image(
                str(retrieval_plot),
                caption="Retrieval Quality Comparison",
                use_container_width=True,
            )

    with col2:
        if safety_plot.exists():
            st.image(
                str(safety_plot),
                caption="Safety and Latency Comparison",
                use_container_width=True,
            )

    if summary_csv.exists():
        st.subheader("Summary Metrics Table")
        df = pd.read_csv(summary_csv)
        st.dataframe(df, use_container_width=True)
