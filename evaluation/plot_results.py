from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluation.summary_utils import (
    MODE_ORDER,
    MODEL_ORDER,
    build_summary,
    load_results,
)


OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
]


def add_value_labels(ax):
    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )


def plot_grouped_metrics(summary, metrics, filename, fig_title):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    bar_width = 0.22
    x = np.arange(len(MODE_ORDER))

    for ax, (metric_col, metric_title) in zip(axes, metrics):
        for i, model_name in enumerate(MODEL_ORDER):
            model_df = (
                summary[summary["model"] == model_name]
                .set_index("mode")
                .reindex(MODE_ORDER)
            )
            values = model_df[metric_col].fillna(0.0).values

            ax.bar(
                x + (i - 1) * bar_width,
                values,
                width=bar_width * 0.9,
                label=model_name,
                color=MODEL_COLORS[i],
                edgecolor="black",
                linewidth=0.7,
            )

        ax.set_title(metric_title, fontsize=13, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(MODE_ORDER, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        add_value_labels(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.10),
        fontsize=11,
    )
    fig.suptitle(fig_title, fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_results("evaluation/results_multimodel.json")
    summary = build_summary(df)

    retrieval_metrics = [
        ("recall_at_5", "Recall@5"),
        ("mrr", "MRR"),
        ("source_hit_rate", "Source Hit Rate"),
    ]
    plot_grouped_metrics(
        summary,
        retrieval_metrics,
        "retrieval_metrics_comparison.png",
        "Retrieval Quality Comparison Across Models",
    )

    safety_metrics = [
        ("leakage_rate", "Leakage Rate"),
        ("policy_success", "Policy Success Rate"),
        ("latency", "Latency (seconds)"),
    ]
    plot_grouped_metrics(
        summary,
        safety_metrics,
        "safety_latency_comparison.png",
        "Safety and Latency Comparison Across Models",
    )

    print(f"Saved plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
