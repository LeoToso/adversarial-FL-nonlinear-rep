"""
sweep.py  —  Grid sweep over all experimental configurations
=============================================================
Iterates over the full Cartesian product of:
  datasets × algorithms × aggregators × attacks ×
  heterogeneity levels × client counts × Byzantine fractions

Results are saved incrementally to JSON so partial runs are preserved.

Usage
-----
python sweep.py                       # full sweep with defaults
python sweep.py --datasets cifar10    # single dataset
python sweep.py --quick               # small grid for debugging
python sweep.py --resume results.json # resume from checkpoint
"""

import os
import sys
import json
import argparse
import itertools
from typing import List, Dict, Any

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_experiment import run_experiment
from evaluation.metrics         import ExperimentResult, save_results
from evaluation.plotting        import generate_all_plots


# ── Default sweep grid ────────────────────────────────────────────────────────

FULL_GRID: Dict[str, List[Any]] = {
    "dataset":     ["cifar10", "cifar100", "femnist", "sent140"],
    "algorithm":   ["baseline", "fedrep_linear", "fedrep_nonlinear"],
    "aggregator":  ["TrMean", "Krum", "GM", "CWMed", "NNM+TrMean", "NNM+GM"],
    "attack":      ["SignFlipping", "InnerProductManipulation", "Zero"],
    "alpha":       [0.1, 0.5, 1.0, 10.0],           # Dirichlet heterogeneity
    "n_clients":   [10, 20, 50],
    "byz_frac":    [0.1, 0.2, 0.3],                  # f/n Byzantine fraction
    "seed":        [42],
}

# Quick grid for debugging / CI
QUICK_GRID: Dict[str, List[Any]] = {
    "dataset":     ["cifar10", "femnist"],
    "algorithm":   ["baseline", "fedrep_nonlinear"],
    "aggregator":  ["TrMean", "NNM+TrMean"],
    "attack":      ["SignFlipping", "Zero"],
    "alpha":       [0.1, 1.0],
    "n_clients":   [10],
    "byz_frac":    [0.2],
    "seed":        [42],
}

# Training hyperparameters per dataset
DATASET_HPS: Dict[str, Dict] = {
    "cifar10":  {"rounds": 300, "batch_size": 64,  "lr": 0.1,  "lr_head": 0.01, "head_steps": 10, "repr_dim": 256},
    "cifar100": {"rounds": 400, "batch_size": 64,  "lr": 0.05, "lr_head": 0.01, "head_steps": 10, "repr_dim": 256},
    "femnist":  {"rounds": 200, "batch_size": 32,  "lr": 0.1,  "lr_head": 0.01, "head_steps": 10, "repr_dim": 128},
    "sent140":  {"rounds": 200, "batch_size": 32,  "lr": 0.05, "lr_head": 0.01, "head_steps": 5,  "repr_dim": 128},
}


def build_configs(grid: Dict[str, List[Any]], device: str) -> List[Dict]:
    """Expand grid into list of experiment configurations."""
    keys = ["dataset", "algorithm", "aggregator", "attack",
            "alpha", "n_clients", "byz_frac", "seed"]
    configs = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(zip(keys, vals))

        # Compute f from fraction
        n = cfg["n_clients"]
        f = max(1, int(cfg["byz_frac"] * n))
        if f >= n / 2:      # skip invalid configs
            continue
        cfg["n_byzantine"] = f
        del cfg["byz_frac"]

        # Add dataset-specific HPs
        hps = DATASET_HPS[cfg["dataset"]]
        cfg.update(hps)
        cfg["device"]    = device
        cfg["eval_every"] = max(1, hps["rounds"] // 50)  # ~50 eval points
        configs.append(cfg)
    return configs


def config_key(cfg: Dict) -> str:
    """Unique string key identifying a configuration (for deduplication)."""
    return (
        f"{cfg['dataset']}_{cfg['algorithm']}_{cfg['aggregator']}"
        f"_{cfg['attack']}_a{cfg['alpha']}_n{cfg['n_clients']}"
        f"_f{cfg['n_byzantine']}_s{cfg['seed']}"
    )


def run_sweep(
    grid:        Dict[str, List[Any]],
    output_dir:  str  = "results",
    device:      str  = "cpu",
    resume_path: str  = None,
    plot:        bool = True,
    verbose:     bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.json")

    # ── Load completed results for resuming ──────────────────────────────────
    completed_keys = set()
    all_results: List[ExperimentResult] = []

    if resume_path and os.path.exists(resume_path):
        with open(resume_path) as f:
            raw = json.load(f)
        for rec in raw:
            all_results.append(rec)         # keep as dict for re-save
            completed_keys.add(config_key(rec))
        print(f"[Resume] Loaded {len(all_results)} completed experiments.")

    # ── Build configs ────────────────────────────────────────────────────────
    configs  = build_configs(grid, device)
    todo     = [c for c in configs if config_key(c) not in completed_keys]
    total    = len(configs)
    n_done   = len(completed_keys)
    n_todo   = len(todo)

    print(f"\n[Sweep] {total} configs total | {n_done} done | {n_todo} to run")
    print(f"        Output: {results_path}\n")

    for idx, cfg in enumerate(todo, start=n_done + 1):
        print(f"\n[{idx}/{total}] {config_key(cfg)}")
        try:
            result = run_experiment(verbose=verbose, **cfg)
            all_results.append(result.to_dict())
        except Exception as e:
            print(f"  [ERROR] {e} — skipping.")
            continue

        # Incremental save after each experiment
        save_results_dicts(all_results, results_path)

    print(f"\n[Sweep] Complete. Results saved to {results_path}")

    if plot and all_results:
        generate_all_plots(results_path, os.path.join(output_dir, "figures"))

    return results_path


def save_results_dicts(results: List[Dict], path: str):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Adversarial FedRep Sweep")
    p.add_argument("--quick",    action="store_true",  help="Use small debug grid")
    p.add_argument("--datasets", nargs="+", default=None,
                   choices=["cifar10","cifar100","femnist","sent140"])
    p.add_argument("--algorithms", nargs="+", default=None,
                   choices=["baseline","fedrep_linear","fedrep_nonlinear"])
    p.add_argument("--aggregators", nargs="+", default=None)
    p.add_argument("--attacks",    nargs="+", default=None)
    p.add_argument("--alphas",     nargs="+", type=float, default=None)
    p.add_argument("--n_clients",  nargs="+", type=int,   default=None)
    p.add_argument("--byz_fracs",  nargs="+", type=float, default=None)
    p.add_argument("--output_dir", default="results")
    p.add_argument("--device",     default="cpu")
    p.add_argument("--resume",     default=None, help="Path to existing results.json")
    p.add_argument("--no_plot",    action="store_true")
    p.add_argument("--verbose",    action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    grid = QUICK_GRID.copy() if args.quick else FULL_GRID.copy()

    # Override grid with CLI arguments
    if args.datasets:    grid["dataset"]    = args.datasets
    if args.algorithms:  grid["algorithm"]  = args.algorithms
    if args.aggregators: grid["aggregator"] = args.aggregators
    if args.attacks:     grid["attack"]     = args.attacks
    if args.alphas:      grid["alpha"]      = args.alphas
    if args.n_clients:   grid["n_clients"]  = args.n_clients
    if args.byz_fracs:   grid["byz_frac"]   = args.byz_fracs

    run_sweep(
        grid=grid,
        output_dir=args.output_dir,
        device=args.device,
        resume_path=args.resume,
        plot=not args.no_plot,
        verbose=args.verbose,
    )
