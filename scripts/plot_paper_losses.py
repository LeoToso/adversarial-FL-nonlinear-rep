#!/usr/bin/env python3
"""Plot per-round training objectives from run_paper_experiments.py."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/paper")
    p.add_argument("--output_dir", default="results/paper_figures")
    args = p.parse_args()

    grouped = defaultdict(list)
    for path in Path(args.results_dir).rglob("seed_*.json"):
        with path.open() as stream:
            record = json.load(stream)
        key = (record["dataset"], record["loss_type"], record["aggregator"],
               record["attack"])
        grouped[key].append(record)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for (dataset, loss_type, aggregator, attack), records in grouped.items():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for algorithm in ("baseline", "fedrep_nonlinear"):
            runs = [r for r in records if r["algorithm"] == algorithm]
            if not runs:
                continue
            common = min(len(r["train_history"]) for r in runs)
            rounds = np.asarray([x["round"] for x in runs[0]["train_history"][:common]])
            losses = np.asarray([
                [x["train_loss"] for x in r["train_history"][:common]] for r in runs
            ])
            mean = losses.mean(axis=0)
            std = losses.std(axis=0)
            label = "Adversarial FL" if algorithm == "baseline" else "Nonlinear FedRep"
            ax.plot(rounds, mean, linewidth=2, label=label)
            ax.fill_between(rounds, mean - std, mean + std, alpha=0.2)
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Training objective")
        ax.set_title(f"{dataset} | {loss_type} | {aggregator} | {attack}")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        stem = f"{dataset}_{loss_type}_{aggregator}_{attack}".replace("+", "plus")
        fig.savefig(output / f"{stem}.png", dpi=180)
        fig.savefig(output / f"{stem}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
