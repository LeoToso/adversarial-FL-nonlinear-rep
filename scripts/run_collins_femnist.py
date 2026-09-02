#!/usr/bin/env python3
"""Run the clean Collins FEMNIST calibration or its Byzantine extension."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.collins_femnist import (CollinsConfig, CollinsFEMNISTTrainer,
                                  load_collins_partition)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", default="data/femnist_collins")
    parser.add_argument("--output_dir", default="results/femnist_collins")
    parser.add_argument("--mode", choices=["clean", "robust"], required=True)
    parser.add_argument("--algorithms", nargs="+", choices=["fedavg", "fedrep"],
                        default=["fedavg", "fedrep"])
    parser.add_argument("--aggregators", nargs="+",
                        default=["NNM+TrMean", "NNM+Krum"])
    parser.add_argument(
        "--attacks", nargs="+",
        choices=["SignFlipping", "InnerProductManipulation", "ALIE"],
        default=["SignFlipping", "InnerProductManipulation"],
    )
    parser.add_argument(
        "--attack_tau", type=float, default=1.5,
        help="Attack factor for IPM and ALIE (default: 1.5)",
    )
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 123, 456, 789, 1024])
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--honest_per_round", type=int, default=15)
    parser.add_argument("--byzantine_per_round", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--standard_logits", action="store_true",
                        help="Use logits with CE instead of the released code's softmax-before-CE")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_path(root: str, config: CollinsConfig) -> Path:
    attack = config.attack if config.byzantine_per_round else "None"
    return (Path(root) / config.algorithm / config.aggregator / attack /
            f"seed_{config.seed}.json")


def main():
    args = parse_args()
    train_sets, test_sets, metadata = load_collins_partition(args.partition)
    if metadata["n_clients"] != 150 or metadata["n_classes"] != 10:
        raise ValueError("This runner requires the 150-client, 10-class Collins partition")

    if args.mode == "clean":
        grid = itertools.product(args.algorithms, ["Average"], ["SignFlipping"], args.seeds)
        byzantine = 0
    else:
        grid = itertools.product(args.algorithms, args.aggregators, args.attacks, args.seeds)
        byzantine = args.byzantine_per_round

    for algorithm, aggregator, attack, seed in grid:
        config = CollinsConfig(
            algorithm=algorithm, rounds=args.rounds,
            honest_per_round=args.honest_per_round,
            byzantine_per_round=byzantine, aggregator=aggregator, attack=attack,
            attack_tau=args.attack_tau,
            eval_every=args.eval_every, seed=seed, device=args.device,
            official_softmax_ce=not args.standard_logits,
        )
        path = output_path(args.output_dir, config)
        checkpoint_path = path.with_suffix(".pt")
        if path.exists() and checkpoint_path.exists() and not args.overwrite:
            print(f"skip {path}")
            continue
        print(f"run {path}", flush=True)
        trainer = CollinsFEMNISTTrainer(config, train_sets, test_sets)
        result = trainer.run()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        os.replace(temporary, path)
        torch.save(trainer.checkpoint(), checkpoint_path)


if __name__ == "__main__":
    main()
