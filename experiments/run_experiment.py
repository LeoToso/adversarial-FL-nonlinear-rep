"""
run_experiment.py  —  Run one (dataset, config) experiment
===========================================================
Trains either baseline adversarial FL or nonlinear adversarial FedRep
for a fixed number of rounds, records per-round metrics, and returns an
ExperimentResult object.
"""

import os
import time
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Optional

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
    algorithm:   str,   # "baseline" | "fedrep_nonlinear"
    aggregator:  str,
    attack:      str,
    loss_type:   str   = "cross_entropy",
    # ── model / training ──────────────────────────────────
    repr_dim:    int   = 256,
    head_steps:  int   = 10,
    lr:          float = 0.1,
    lr_head:     float = 0.01,
    lr_schedule: str   = "constant",
    lr_min:      float = 0.0,
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
    global_probe: bool = False,
    checkpoint_path: str = None,
    head_optimizer: str = "sgd",
) -> ExperimentResult:
    """
    Full training loop for one experiment configuration.

    Returns an ExperimentResult with per-round accuracy/loss history.
    """
    set_seed(seed)
    attack_kwargs = attack_kwargs or {}
    task = DATASET_META[dataset].get("task", "classification")
    if lr_schedule not in {"constant", "cosine"}:
        raise ValueError(
            f"Unknown lr_schedule '{lr_schedule}'; expected constant or cosine."
        )
    if not 0.0 <= lr_min <= lr:
        raise ValueError("lr_min must satisfy 0 <= lr_min <= lr")

    def server_learning_rate(round_number: int) -> float:
        """Learning rate used by the shared server parameter at this round."""
        if lr_schedule == "constant" or rounds <= 1:
            return float(lr)
        progress = (round_number - 1) / (rounds - 1)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return float(lr_min + (lr - lr_min) * cosine)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Dataset: {dataset}  |  α={alpha}  |  n={n_clients}  |  f={n_byzantine}")
        print(f"  Algorithm: {algorithm}  |  Loss: {loss_type}  |  "
              f"Aggregator: {aggregator}  |  Attack: {attack}")
        print(f"{'='*70}")

    # ── Data ─────────────────────────────────────────────────────────────────
    device_obj = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    if verbose:
        print(f"  [1/4] Downloading / loading {dataset} data ...", flush=True)

    n_hon = n_clients - n_byzantine
    client_loaders, client_test_loaders, test_loader = get_loaders(
        dataset, n_hon, alpha,
        data_dir=data_dir, batch_size=batch_size, seed=seed,
        use_leaf=use_leaf
    )
    if len(client_loaders) != n_hon or len(client_test_loaders) != n_hon:
        raise ValueError(
            f"Expected {n_hon} honest client loaders, received "
            f"{len(client_loaders)} train and {len(client_test_loaders)} test."
        )
    meta = DATASET_META[dataset]

    if verbose:
        sizes = [len(ld.dataset) for ld in client_loaders]
        print(f"        {n_hon} honest clients | "
              f"samples/client: min={min(sizes)} max={max(sizes)}")
        print(f"  [2/4] Computing heterogeneity score ...", flush=True)

    het_score = heterogeneity_score(client_loaders, meta["n_classes"], task=task)

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

    if algorithm == "baseline":
        trainer = FedBaseline(
            dataset=dataset, n_clients=n_clients, n_byzantine=n_byzantine,
            aggregator=agg, attack=atk, repr_dim=repr_dim,
            lr=lr, momentum=momentum, linear=False,
            device=str(device_obj), loss_type=loss_type,
        )
        trainer.client_test_loaders = client_test_loaders
    elif algorithm == "fedrep_nonlinear":
        trainer = FedRep(
            dataset=dataset, n_clients=n_clients, n_byzantine=n_byzantine,
            aggregator=agg, attack=atk, repr_dim=repr_dim,
            lr_backbone=lr, lr_head=lr_head, head_steps=head_steps,
            head_optimizer=head_optimizer,
            momentum=momentum, linear=False,
            device=str(device_obj), loss_type=loss_type,
        )
        trainer.client_loaders      = client_loaders
        trainer.client_test_loaders = client_test_loaders
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'.")

    # ── Result object ──────────────────────────────────────────────────────────
    result = ExperimentResult(
        dataset=dataset, n_clients=n_clients, n_byzantine=n_byzantine,
        alpha=alpha, aggregator=aggregator, attack=attack,
        algorithm=algorithm, repr_dim=repr_dim, head_steps=head_steps, seed=seed,
        loss_type=loss_type, task=task,
        het_score=het_score,
    )

    timer = Timer()
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

        current_lr = server_learning_rate(rnd)
        if algorithm == "baseline":
            trainer.lr = current_lr
        else:
            trainer.lr_backbone = current_lr

        train_info = trainer.train_round(client_loaders)
        if not np.isfinite(train_info["loss"]):
            raise FloatingPointError(
                f"Non-finite training loss at round {rnd}: "
                f"{train_info['loss']}"
            )
        result.add_train(
            rnd, train_info["loss"], timer.elapsed(),
            learning_rate=current_lr,
        )

        if rnd % eval_every == 0 or rnd == rounds:
            if algorithm == "baseline":
                eval_info = trainer.evaluate(
                    test_loader,
                    client_loaders=client_loaders,
                    head_steps=head_steps,
                    lr_head=lr_head,
                )
                global_info = (trainer.evaluate_global(test_loader)
                               if global_probe else eval_info)
            else:
                eval_info = trainer.evaluate(test_loader)
                global_info = (trainer.evaluate_global(test_loader)
                               if global_probe else eval_info)

            finite_values = {
                "test_loss": eval_info.get("test_loss"),
                "metric_value": eval_info.get("metric_value"),
                "global_loss": global_info.get("test_loss"),
                "global_metric_value": global_info.get("metric_value"),
            }
            for name, value in finite_values.items():
                if value is not None and not np.isfinite(value):
                    raise FloatingPointError(
                        f"Non-finite {name} at evaluation round {rnd}: {value}"
                    )

            rec = RoundRecord(
                round      = rnd,
                train_loss = train_info["loss"],
                test_acc   = eval_info.get("test_acc"),
                test_loss  = eval_info["test_loss"],
                wall_time  = timer.elapsed(),
                global_acc = global_info.get("test_acc"),
                metric_name = eval_info["metric_name"],
                metric_value = eval_info["metric_value"],
                global_loss = global_info.get("test_loss"),
                global_metric_value = global_info.get("metric_value"),
                learning_rate = current_lr,
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
                    f"{rec.metric_name}={rec.metric_value:.4f}  |  "
                    f"global_loss={global_info['test_loss']:.4f}  |  "
                    f"t={rec.wall_time:.0f}s  ETA {eta_str}"
                )

    if verbose:
        print(f"\n  ✓ Done — best {result.history[-1].metric_name}="
              f"{result.best_metric:.4f}, final={result.final_metric:.4f}\n")

    if checkpoint_path is not None:
        def cpu_state(module):
            return {name: value.detach().cpu()
                    for name, value in module.state_dict().items()}

        checkpoint = {
            "format_version": 1,
            "round": rounds,
            "config": {
                "dataset": dataset,
                "alpha": alpha,
                "n_clients": n_clients,
                "n_byzantine": n_byzantine,
                "algorithm": algorithm,
                "aggregator": aggregator,
                "attack": attack,
                "loss_type": loss_type,
                "repr_dim": repr_dim,
                "head_steps": head_steps,
                "lr": lr,
                "lr_head": lr_head,
                "head_optimizer": head_optimizer if algorithm != "baseline" else None,
                "lr_schedule": lr_schedule,
                "lr_min": lr_min,
                "momentum": momentum,
                "rounds": rounds,
                "batch_size": batch_size,
                "seed": seed,
            },
            "result_summary": {
                "best_metric": result.best_metric,
                "final_metric": result.final_metric,
                "metric_name": result.history[-1].metric_name,
            },
        }
        if algorithm == "baseline":
            checkpoint["model_state_dict"] = cpu_state(trainer.model)
        else:
            checkpoint["backbone_state_dict"] = cpu_state(trainer.backbone)
            checkpoint["client_head_state_dicts"] = [
                cpu_state(model.head) for model in trainer.client_models
            ]

        checkpoint_path = os.path.abspath(checkpoint_path)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        temporary_checkpoint = checkpoint_path + ".tmp"
        torch.save(checkpoint, temporary_checkpoint)
        os.replace(temporary_checkpoint, checkpoint_path)

    return result
