from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


MODEL_ORDER = [
    "gpt-5.4-2026-03-05",
    "gpt-5.1-2025-11-13",
    "gpt-4o-2024-11-20",
]

MODE_ORDER = [
    "naive_llm",
    "plain_rag",
    "sentinelrag",
]


def load_results(
    path: str | Path = "evaluation/results_multimodel.json",
) -> pd.DataFrame:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    benign = df[df["type"] == "benign"]
    adv = df[df["type"] == "adversarial"]

    summary = df.groupby(["model", "mode"], as_index=False).agg(
        latency=("latency", "mean"),
        leakage_rate=("leakage", "mean"),
    )

    benign_summary = benign.groupby(["model", "mode"], as_index=False).agg(
        recall_at_5=("recall_at_5", "mean"),
        mrr=("mrr", "mean"),
        source_hit_rate=("source_hit_rate", "mean"),
    )

    adv_summary = adv.groupby(["model", "mode"], as_index=False).agg(
        policy_success=("policy_success", "mean"),
    )

    summary = summary.merge(benign_summary, on=["model", "mode"], how="left")
    summary = summary.merge(adv_summary, on=["model", "mode"], how="left")

    summary["model"] = pd.Categorical(
        summary["model"], categories=MODEL_ORDER, ordered=True
    )
    summary["mode"] = pd.Categorical(
        summary["mode"], categories=MODE_ORDER, ordered=True
    )

    return summary.sort_values(["model", "mode"]).reset_index(drop=True)
