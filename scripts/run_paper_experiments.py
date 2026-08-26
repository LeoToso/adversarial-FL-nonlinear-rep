#!/usr/bin/env python3
"""Run the experiment grid specified in the ICLR 2027 paper draft.

Each configuration is written to its own JSON file so jobs can be resumed or
distributed without corrupting a shared results file.  Every JSON contains
``train_history`` (one loss value per communication round) and ``history``
(held-out metrics at the requested evaluation cadence).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AGGREGATORS = ["NNM+TrMean", "NNM+GM", "NNM+Krum"]
ATTACKS = ["SignFlipping", "InnerProductManipulation"]
ALGORITHMS = ["baseline", "fedrep_nonlinear"]
LOSSES = {
    "cifar10": ["cross_entropy", "multiclass_ls"],
    "femnist": ["cross_entropy", "multiclass_ls"],
    "school": ["least_squares"],
}
DEFAULTS = {
    "cifar10": dict(n_clients=15, n_byzantine=5, alpha=0.1, rounds=300,
                    batch_size=64, repr_dim=256, head_steps=10,
                    lr=0.1, lr_head=0.01, use_leaf=False),
    "femnist": dict(n_clients=15, n_byzantine=5, alpha=1.0, rounds=200,
                    batch_size=32, repr_dim=128, head_steps=10,
                    lr=0.1, lr_head=0.01, use_leaf=True),
    # 139 honest school tasks plus 35 Byzantine workers (~20% Byzantine).
    "school": dict(n_clients=174, n_byzantine=35, alpha=1.0, rounds=500,
                   batch_size=32, repr_dim=64, head_steps=20,
                   lr=0.05, lr_head=0.01, use_leaf=False),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(LOSSES),
                   choices=list(LOSSES))
    p.add_argument("--seeds", nargs="+", type=int,
                   default=[42, 123, 456, 789, 1024])
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--output_dir", default="results/paper")
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--rounds", type=int, default=None,
                   help="Override the dataset-specific number of rounds")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def result_path(root, cfg):
    return (Path(root) / cfg["dataset"] / cfg["loss_type"] /
            cfg["algorithm"] / cfg["aggregator"] / cfg["attack"] /
            f"seed_{cfg['seed']}.json")


def main():
    args = parse_args()
    run_experiment = None
    configs = []
    for dataset in args.datasets:
        for loss, algorithm, aggregator, attack, seed in itertools.product(
            LOSSES[dataset], ALGORITHMS, AGGREGATORS, ATTACKS, args.seeds
        ):
            cfg = dict(DEFAULTS[dataset])
            cfg.update(dataset=dataset, loss_type=loss, algorithm=algorithm,
                       aggregator=aggregator, attack=attack, seed=seed,
                       data_dir=args.data_dir, device=args.device,
                       eval_every=args.eval_every)
            if args.rounds is not None:
                cfg["rounds"] = args.rounds
            configs.append(cfg)

    print(f"Prepared {len(configs)} configurations.")
    for index, cfg in enumerate(configs, 1):
        path = result_path(args.output_dir, cfg)
        label = (f"{cfg['dataset']}/{cfg['loss_type']}/{cfg['algorithm']}/"
                 f"{cfg['aggregator']}/{cfg['attack']}/seed={cfg['seed']}")
        if path.exists() and not args.overwrite:
            print(f"[{index}/{len(configs)}] skip {label}")
            continue
        print(f"[{index}/{len(configs)}] run  {label}")
        if args.dry_run:
            continue
        if run_experiment is None:
            from experiments.run_experiment import run_experiment as _run_experiment
            run_experiment = _run_experiment
        result = run_experiment(**cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(result.to_dict(), stream, indent=2, allow_nan=False)
        os.replace(temporary, path)


if __name__ == "__main__":
    main()
