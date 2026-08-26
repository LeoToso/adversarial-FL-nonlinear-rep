#!/usr/bin/env python3
"""
main.py  —  Adversarial Robust Federated Representation Learning
================================================================
Single entry point for both individual experiments and full sweeps.

Examples
--------
# Single experiment
python main.py --dataset cifar10 --alpha 0.1 --n_clients 10 --n_byzantine 2 \
               --algorithm fedrep_nonlinear --aggregator NNM+TrMean \
               --attack SignFlipping --rounds 200

# Compare all three algorithms on one config
python main.py --dataset cifar10 --alpha 0.5 --n_clients 10 --n_byzantine 2 \
               --mode all --aggregator NNM+TrMean --attack SignFlipping

# Full sweep  (all datasets × algorithms × configs)
python main.py --sweep

# Quick debug sweep
python main.py --sweep --quick

# Resume interrupted sweep
python main.py --sweep --resume results/results.json

# Only plot from existing results
python main.py --plot_only --results results/results.json
"""

import os
import sys
import argparse
import json

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.run_experiment import run_experiment
from experiments.sweep          import run_sweep, FULL_GRID, QUICK_GRID, save_results_dicts
from evaluation.metrics         import ExperimentResult
from evaluation.plotting        import generate_all_plots


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Adversarial FL vs Adversarial FedRep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--sweep",     action="store_true", help="Run full grid sweep")
    mode.add_argument("--plot_only", action="store_true", help="Only generate plots")

    # Data
    p.add_argument("--dataset",    default="cifar10",
                   choices=["cifar10", "cifar100", "femnist", "sent140",
                            "heart_disease", "isic2019", "school"])
    p.add_argument("--data_dir",   default="./data")
    p.add_argument("--alpha",      type=float, default=0.5,
                   help="Dirichlet alpha (heterogeneity)")
    p.add_argument("--batch_size", type=int,   default=64)

    p.add_argument("--use_leaf",   action="store_true", default=False,
               help="Use LEAF natural heterogeneity for FEMNIST")

    # Federated
    p.add_argument("--n_clients",   type=int, default=10)
    p.add_argument("--n_byzantine", type=int, default=2)

    # Algorithm
    p.add_argument("--algorithm", default="fedrep_nonlinear",
                   choices=["baseline", "fedrep_linear", "fedrep_nonlinear", "local", "all"],
                   help="'all' runs baseline + both fedrep variants")
    p.add_argument("--repr_dim",   type=int, default=256)
    p.add_argument("--head_steps", type=int, default=10)
    p.add_argument("--loss_type", default=None,
                   choices=["cross_entropy", "multiclass_ls", "least_squares"],
                   help="Defaults to least_squares for School and cross_entropy otherwise")

    # Aggregator / attack
    p.add_argument("--aggregator", default="NNM+TrMean",
                   choices=["Average","TrMean","Krum","GM","CWMed",
                             "NNM+TrMean","NNM+GM","NNM+Krum","NNM+CWMed"])
    p.add_argument("--attack", default="SignFlipping",
                   choices=["SignFlipping","InnerProductManipulation",
                             "FallOfEmpires","Mimic","Zero","Random"])

    # Training
    p.add_argument("--rounds",    type=int,   default=200)
    p.add_argument("--lr",        type=float, default=0.1)
    p.add_argument("--lr_head",   type=float, default=0.01)
    p.add_argument("--momentum",  type=float, default=0.9)
    p.add_argument("--eval_every",type=int,   default=5)

    # Sweep-specific
    p.add_argument("--quick",  action="store_true", help="Small debug grid")
    p.add_argument("--resume", default=None,        help="Resume from results.json")

    # Misc
    p.add_argument("--device",     default="cpu")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output_dir", default="results")
    p.add_argument("--results",    default=None,
                   help="Path to results.json for --plot_only")
    p.add_argument("--no_plot",    action="store_true")
    p.add_argument("--quiet",      action="store_true")

    return p


# ── Helpers ───────────────────────────────────────────────────────────────────

def single_run(args, algorithm: str) -> ExperimentResult:
    from core.datasets import DATASET_META
    meta = DATASET_META[args.dataset]

    # Auto-set repr_dim per dataset if user left default
    repr_dim = args.repr_dim
    head_steps = args.head_steps

    return run_experiment(
        dataset=args.dataset,
        alpha=args.alpha,
        n_clients=args.n_clients,
        n_byzantine=args.n_byzantine,
        algorithm=algorithm,
        aggregator=args.aggregator,
        attack=args.attack,
        loss_type=(args.loss_type or
                   ("least_squares" if args.dataset == "school" else "cross_entropy")),
        repr_dim=repr_dim,
        head_steps=head_steps,
        lr=args.lr,
        lr_head=args.lr_head,
        momentum=args.momentum,
        rounds=args.rounds,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        device=args.device,
        seed=args.seed,
        eval_every=args.eval_every,
        verbose=not args.quiet,
        use_leaf=args.use_leaf,
    )


def print_summary(results):
    print("\n" + "="*60)
    print(f"{'Algorithm':<25} {'Best metric':>12} {'Final metric':>12}")
    print("-"*60)
    for r in results:
        print(f"  {r.algorithm:<23} {r.best_metric:>10.4f}  {r.final_metric:>10.4f}")
    print("="*60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Plot-only mode ────────────────────────────────────────────────────────
    if args.plot_only:
        path = args.results or os.path.join(args.output_dir, "results.json")
        if not os.path.exists(path):
            print(f"[ERROR] Results file not found: {path}")
            sys.exit(1)
        generate_all_plots(path, os.path.join(args.output_dir, "figures"))
        return

    # ── Sweep mode ────────────────────────────────────────────────────────────
    if args.sweep:
        grid = QUICK_GRID.copy() if args.quick else FULL_GRID.copy()
        run_sweep(
            grid=grid,
            output_dir=args.output_dir,
            device=args.device,
            resume_path=args.resume,
            plot=not args.no_plot,
            verbose=not args.quiet,
        )
        return

    # ── Single / multi-algorithm run ─────────────────────────────────────────
    algorithms = (
        ["baseline", "fedrep_linear", "fedrep_nonlinear"]
        if args.algorithm == "all"
        else [args.algorithm]
    )

    all_results = []
    for algo in algorithms:
        result = single_run(args, algo)
        all_results.append(result)

    print_summary(all_results)

    # Save results
    out_path = os.path.join(args.output_dir, "results.json")
    # Merge with existing if present
    existing = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    combined = existing + [r.to_dict() for r in all_results]
    save_results_dicts(combined, out_path)
    print(f"Results saved to {out_path}")

    # Quick plot for single run
    if not args.no_plot:
     pass  # removed auto-plotting — use --plot_only explicitly


if __name__ == "__main__":
    main()
