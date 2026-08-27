#!/usr/bin/env python3
"""Create per-seed and mean ± standard-deviation paper result tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from scipy.stats import t as student_t


GROUP_COLUMNS = [
    "dataset", "loss_type", "algorithm", "aggregator", "attack",
    "metric_name",
]


def load_rows(results_dir: Path):
    rows = []
    for path in sorted(results_dir.rglob("seed_*.json")):
        with path.open() as stream:
            record = json.load(stream)
        if not record.get("history") or not record.get("train_history"):
            continue
        final_eval = record["history"][-1]
        final_train = record["train_history"][-1]
        rows.append({
            "dataset": record["dataset"],
            "loss_type": record["loss_type"],
            "algorithm": record["algorithm"],
            "aggregator": record["aggregator"],
            "attack": record["attack"],
            "seed": int(record["seed"]),
            "metric_name": final_eval["metric_name"],
            "final_metric": float(final_eval["metric_value"]),
            "best_metric": float(record["best_metric"]),
            "final_test_loss": float(final_eval["test_loss"]),
            "final_train_loss": float(final_train["train_loss"]),
            "wall_time_seconds": float(final_eval["wall_time"]),
            "heterogeneity_score": float(record["het_score"]),
            "result_file": str(path),
            "checkpoint_file": str(path.with_suffix(".pt")),
            "checkpoint_exists": path.with_suffix(".pt").exists(),
        })
    return rows


def plus_minus(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/paper")
    parser.add_argument("--output_dir", default="results/paper_tables")
    parser.add_argument("--expected_seeds", nargs="+", type=int,
                        default=[42, 123, 456, 789, 1024])
    args = parser.parse_args()

    rows = load_rows(Path(args.results_dir))
    if not rows:
        raise SystemExit(f"No result JSON files found under {args.results_dir}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame(rows).sort_values(GROUP_COLUMNS + ["seed"])
    per_seed.to_csv(output / "per_seed_results.csv", index=False)

    numeric = [
        "final_metric", "best_metric", "final_test_loss",
        "final_train_loss", "wall_time_seconds", "heterogeneity_score",
    ]
    grouped = per_seed.groupby(GROUP_COLUMNS, dropna=False)
    summary = grouped[numeric].agg(["mean", "std"]).reset_index()
    summary.columns = [
        column if isinstance(column, str) else
        column[0] if not column[1] else f"{column[0]}_{column[1]}"
        for column in summary.columns
    ]
    counts = grouped["seed"].nunique().rename("n_seeds").reset_index()
    summary = summary.merge(counts, on=GROUP_COLUMNS, how="left")
    for metric in numeric:
        summary[f"{metric}_pm_std"] = [
            plus_minus(mean, std)
            for mean, std in zip(summary[f"{metric}_mean"],
                                 summary[f"{metric}_std"])
        ]
        half_widths = []
        for std, count in zip(summary[f"{metric}_std"], summary["n_seeds"]):
            half_widths.append(
                float(student_t.ppf(0.975, count - 1) * std / math.sqrt(count))
                if count > 1 else float("nan")
            )
        summary[f"{metric}_ci95_half_width"] = half_widths
        summary[f"{metric}_pm_ci95"] = [
            f"{mean:.4f} ± {half_width:.4f}"
            for mean, half_width in zip(summary[f"{metric}_mean"], half_widths)
        ]
    summary.to_csv(output / "mean_std_results.csv", index=False)

    display_columns = GROUP_COLUMNS + [
        "n_seeds", "final_metric_pm_std", "best_metric_pm_std",
        "final_test_loss_pm_std", "final_train_loss_pm_std",
        "final_metric_pm_ci95",
    ]
    with (output / "mean_std_results.md").open("w") as stream:
        stream.write("| " + " | ".join(display_columns) + " |\n")
        stream.write("| " + " | ".join(["---"] * len(display_columns)) + " |\n")
        for _, row in summary[display_columns].iterrows():
            stream.write("| " + " | ".join(str(row[column])
                         for column in display_columns) + " |\n")

    expected = set(args.expected_seeds)
    incomplete = []
    for key, group in grouped:
        missing = sorted(expected - set(group["seed"]))
        missing_checkpoints = int((~group["checkpoint_exists"]).sum())
        if missing or missing_checkpoints:
            incomplete.append((*key, missing, missing_checkpoints))
    if incomplete:
        print("WARNING: incomplete configurations:")
        for item in incomplete:
            print(item)
    else:
        print("All configurations contain every expected seed and checkpoint.")
    print(f"Wrote tables to {output}")


if __name__ == "__main__":
    main()
