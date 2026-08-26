"""
run_experiment.py  —  Run one (dataset, config) experiment
===========================================================
Trains either the baseline, fedrep_linear, or fedrep_nonlinear algorithm
for a fixed number of rounds, records per-round metrics, and returns an
ExperimentResult object.
"""

import time
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Optional
from core.fed_local import LocalOnly

from core.datasets    import get_loaders, DATASET_META, heterogeneity_score
from core.aggregators import RobustAggregator, ByzantineAttack
from core.fed_baseline import FedBaseline
from core.fed_fedrep   import FedRep
from evaluation.metrics import ExperimentResult, RoundRecord, Timer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_experiment(
    # ── data ──────────────────────────────────────────────
    dataset:     str,
    alpha:       float,
    n_clients:   int,
    n_byzantine: int,
    # ── algorithm ─────────────────────────────────────────
    algorithm:   str,   # "baseline" | "fedrep_linear" | "fedrep_nonlinear"
    aggregator:  str,
    attack:      str,
    # ── model / training ──────────────────────────────────
    repr_dim:    int   = 256,
    head_steps:  int   = 10,
    lr:          float = 0.1,
    lr_head:     float = 0.01,
    momentum:    float = 0.9,
    rounds:      int   = 200,
    batch_size:  int   = 64,
    # ── misc ──────────────────────────────────────────────
    data_dir:    str   = "./data",
    device:      str   = "cpu",
    seed:        int   = 42,
    eval_every:  int   = 5,
    attack_kwargs: dict = None,
    verbose:     bool  = True,
    use_leaf:    bool  = False,
) -> ExperimentResult:
    """
    Full training loop for one experiment configuration.

    Returns an ExperimentResult with per-round accuracy/loss history.
    """
    set_seed(seed)
    attack_kwargs = attack_kwargs or {}
    meta = DATASET_META[dataset]

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Dataset: {dataset}  |  α={alpha}  |  n={n_clients}  |  f={n_byzantine}")
        print(f"  Algorithm: {algorithm}  |  Aggregator: {aggregator}  |  Attack: {attack}")
        print(f"{'='*70}")

    # ── Data ─────────────────────────────────────────────────────────────────
    device_obj = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    if verbose:
        print(f"  [1/4] Downloading / loading {dataset} data ...", flush=True)

    client_loaders, client_test_loaders, test_loader = get_loaders(
        dataset, n_clients, alpha,
        data_dir=data_dir, batch_size=batch_size, seed=seed,
        use_leaf=use_leaf
    )

    if verbose:
        sizes = [len(ld.dataset) for ld in client_loaders]
        print(f"        {n_clients - n_byzantine} honest clients | "
              f"samples/client: min={min(sizes)} max={max(sizes)}")
        print(f"  [2/4] Computing heterogeneity score ...", flush=True)

    het_score = heterogeneity_score(client_loaders, meta["n_classes"])

    if verbose:
        print(f"        Heterogeneity score: {het_score:.4f}  "
              f"(0=IID, 2=extreme non-IID)")

    # ── Aggregator / attack ───────────────────────────────────────────────────
    if verbose:
        print(f"  [3/4] Setting up aggregator ({aggregator}) and "
              f"attack ({attack}) ...", flush=True)

    agg = RobustAggregator(aggregator, n_byzantine)
    atk = ByzantineAttack(attack, n_byzantine, **attack_kwargs)

    # ── Algorithm ─────────────────────────────────────────────────────────────
    if verbose:
        print(f"  [4/4] Building model ({algorithm}) ...", flush=True)

    linear = (algorithm == "fedrep_linear")

    if algorithm == "baseline":
        trainer = FedBaseline(
            dataset=dataset, n_clients=n_clients, n_byzantine=n_byzantine,
            aggregator=agg, attack=atk, repr_dim=repr_dim,
            lr=lr, momentum=momentum, linear=False,
            device=str(device_obj),
        )
        trainer.client_test_loaders = client_test_loaders
    elif algorithm in ("fedrep_linear", "fedrep_nonlinear"):
        trainer = FedRep(
            dataset=dataset, n_clients=n_clients, n_byzantine=n_byzantine,
            aggregator=agg, attack=atk, repr_dim=repr_dim,
            lr_backbone=lr, lr_head=lr_head, head_steps=head_steps,
            momentum=momentum, linear=linear,
            device=str(device_obj),
        )
        trainer.client_loaders = client_loaders
        trainer.client_loaders      = client_loaders
        trainer.client_test_loaders = client_test_loaders
    elif algorithm == "local":
        from core.models import build_model
        init_model = build_model(dataset, repr_dim, meta["n_classes"], linear=False).to(device_obj)
        trainer = LocalOnly(
            init_model, client_loaders, client_test_loaders,
            lr=lr, momentum=momentum, device=device_obj
    )
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'.")

    # ── Result object ──────────────────────────────────────────────────────────
    result = ExperimentResult(
        dataset=dataset, n_clients=n_clients, n_byzantine=n_byzantine,
        alpha=alpha, aggregator=aggregator, attack=attack,
        algorithm=algorithm, repr_dim=repr_dim, head_steps=head_steps, seed=seed,
    )

    timer = Timer()
    n_hon = n_clients - n_byzantine

    if verbose:
        print(f"\n  Starting training: {rounds} rounds, "
              f"evaluating every {eval_every} rounds")
        print(f"  {'Round':>6}  {'Loss':>8}  {'Test Acc':>9}  {'Time':>6}")
        print(f"  {'-'*38}")

    # ── Training loop ─────────────────────────────────────────────────────────
    for rnd in range(1, rounds + 1):

        # Print a heartbeat every round so user sees progress
        if verbose and rnd % max(1, eval_every // 2) == 0:
            print(f"  round {rnd:4d}/{rounds} — training ...", end="\r", flush=True)

        train_info = trainer.train_round(client_loaders)

        early_eval = rnd <= 500 and rnd % 10 == 0
        late_eval  = rnd > 500 and rnd % eval_every == 0
        if early_eval or late_eval or rnd == rounds:
            if algorithm == "baseline":
                eval_info = trainer.evaluate(
                    test_loader,
                    client_loaders=client_loaders,
                    head_steps=head_steps,
                    lr_head=lr_head,
                )
                global_info = trainer.evaluate(test_loader)
            else:
                eval_info = trainer.evaluate(test_loader)
                global_info = trainer.evaluate_global(test_loader)

            rec = RoundRecord(
                round      = rnd,
                train_loss = train_info["loss"],
                test_acc   = eval_info["test_acc"],
                test_loss  = eval_info["test_loss"],
                wall_time  = timer.elapsed(),
                global_acc = global_info["test_acc"],   # ← add this
            )
            result.add(rec)

            if verbose:
                bar_len = 20
                filled  = int(bar_len * rnd / rounds)
                bar     = "█" * filled + "░" * (bar_len - filled)
                eta_s   = (rec.wall_time / rnd) * (rounds - rnd)
                eta_str = f"{int(eta_s//60)}m{int(eta_s%60):02d}s"
                print(
                    f"  [{bar}] {rnd:4d}/{rounds}  |  "
                    f"loss={rec.train_loss:.3f}  |  "
                    f"pers={rec.test_acc*100:5.1f}%  |  "
                    f"global={global_info['test_acc']*100:5.1f}%  |  "
                    f"t={rec.wall_time:.0f}s  ETA {eta_str}"
                )

    if verbose:
        print(f"\n  ✓ Done — best={result.best_acc*100:.2f}%  "
              f"final={result.final_acc*100:.2f}%\n")

    return result
