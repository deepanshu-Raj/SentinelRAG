from __future__ import annotations

from pathlib import Path

from evaluation.summary_utils import build_summary, load_results


OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = load_results("evaluation/results_multimodel.json")
    summary = build_summary(df)

    summary.to_csv(OUT_DIR / "summary_metrics.csv", index=False)

    for model_name in summary["model"].cat.categories:
        model_df = summary[summary["model"] == model_name]
        if model_df.empty:
            continue

        print(f"\n########## MODEL: {model_name} ##########\n")
        for _, row in model_df.iterrows():
            print(f"=== {row['mode']} ===")
            print(f"Recall@5:            {row['recall_at_5']:.3f}")
            print(f"MRR:                 {row['mrr']:.3f}")
            print(f"Source Hit Rate:     {row['source_hit_rate']:.3f}")
            print(f"Latency:             {row['latency']:.3f}s")
            print(f"Leakage Rate:        {row['leakage_rate']:.3f}")
            print(f"Policy Success Rate: {row['policy_success']:.3f}")
            print()

    print(f"Saved summary CSV to {OUT_DIR / 'summary_metrics.csv'}")


if __name__ == "__main__":
    main()
