"""Collins et al. (ICML 2021) FEMNIST benchmark and robust extension.

This module intentionally keeps the benchmark's *population* of persistent
client heads separate from the messages received in a communication round.
The clean reference samples 15 of 150 clients.  The robust extension samples
the same 15 honest clients and appends ``f`` Byzantine messages.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from core.aggregators import ByzantineAttack, RobustAggregator


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32)).reshape(-1, 1, 28, 28)
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


def load_collins_partition(path: str | os.PathLike) -> Tuple[List[Dataset], List[Dataset], dict]:
    """Load a partition produced by ``prepare_collins_femnist.py``."""
    root = Path(path)
    archive = np.load(root / "partition.npz")
    with (root / "metadata.json").open() as stream:
        metadata = json.load(stream)

    train, test = [], []
    tr_off = archive["train_offsets"]
    te_off = archive["test_offsets"]
    for client in range(metadata["n_clients"]):
        train.append(ArrayDataset(
            archive["train_x"][tr_off[client]:tr_off[client + 1]],
            archive["train_y"][tr_off[client]:tr_off[client + 1]],
        ))
        test.append(ArrayDataset(
            archive["test_x"][te_off[client]:te_off[client + 1]],
            archive["test_y"][te_off[client]:te_off[client + 1]],
        ))
    return train, test, metadata


class CollinsMLP(nn.Module):
    """Official 784-512-256-64 representation and local 64-10 head."""

    def __init__(self, official_softmax_ce: bool = True):
        super().__init__()
        self.representation = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
        )
        self.head = nn.Linear(64, 10)
        self.official_softmax_ce = official_softmax_ce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.representation(x.flatten(1)))
        # The released ICML 2021 implementation applies softmax before
        # CrossEntropyLoss.  Preserve it by default for numerical replication,
        # while allowing a conventional logits-only ablation.
        return torch.softmax(logits, dim=1) if self.official_softmax_ce else logits


def _flat(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in parameters])


def _set_flat(parameters: Iterable[torch.Tensor], vector: torch.Tensor) -> None:
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.data.copy_(vector[offset:offset + count].view_as(parameter))
        offset += count
    if offset != vector.numel():
        raise ValueError("Flat vector has the wrong number of elements")


def _optimizer(model: nn.Module, lr: float, momentum: float):
    weights, biases = [], []
    for name, parameter in model.named_parameters():
        (biases if "bias" in name else weights).append(parameter)
    return torch.optim.SGD(
        [{"params": weights, "weight_decay": 1e-4},
         {"params": biases, "weight_decay": 0.0}],
        lr=lr, momentum=momentum,
    )


def _set_trainable(model: CollinsMLP, *, representation: bool, head: bool) -> None:
    for parameter in model.representation.parameters():
        parameter.requires_grad_(representation)
    for parameter in model.head.parameters():
        parameter.requires_grad_(head)


def _local_epochs(
    model: CollinsMLP,
    dataset: Dataset,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    if epochs == 0:
        return math.nan
    model.train()
    total_loss = total_samples = 0
    for _ in range(epochs):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            generator=generator, num_workers=0)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            total_samples += len(y)
    return total_loss / total_samples


@torch.no_grad()
def _evaluate(model: CollinsMLP, dataset: Dataset, batch_size: int,
              criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss_sum += criterion(output, y).item() * len(y)
        correct += (output.argmax(dim=1) == y).sum().item()
        total += len(y)
    return correct / total, loss_sum / total


@dataclass(frozen=True)
class CollinsConfig:
    algorithm: str = "fedrep"
    rounds: int = 200
    population_clients: int = 150
    honest_per_round: int = 15
    byzantine_per_round: int = 0
    aggregator: str = "Average"
    attack: str = "SignFlipping"
    attack_tau: float = 1.5
    mimic_client: int = 0
    batch_size: int = 10
    lr: float = 0.01
    momentum: float = 0.5
    head_epochs: int = 10
    representation_epochs: int = 5
    eval_every: int = 10
    seed: int = 42
    device: str = "cpu"
    official_softmax_ce: bool = True


class CollinsFEMNISTTrainer:
    """Clean FedRep reproduction plus an explicitly labelled robust extension."""

    def __init__(self, config: CollinsConfig, train_sets: Sequence[Dataset],
                 test_sets: Sequence[Dataset]):
        self.cfg = config
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if config.algorithm not in {"fedrep", "fedavg"}:
            raise ValueError("algorithm must be fedrep or fedavg")
        if len(train_sets) != config.population_clients or len(test_sets) != config.population_clients:
            raise ValueError("partition size does not match population_clients")
        if not 1 <= config.honest_per_round <= config.population_clients:
            raise ValueError("invalid honest_per_round")
        if config.byzantine_per_round == 0 and config.aggregator != "Average":
            raise ValueError("clean Collins replication must use Average")
        if config.byzantine_per_round > 0 and config.aggregator == "Average":
            raise ValueError("robust extension requires a robust aggregator")

        self.train_sets, self.test_sets = list(train_sets), list(test_sets)
        self.device = torch.device(config.device)
        self.global_model = CollinsMLP(config.official_softmax_ce).to(self.device)
        initial_head = copy.deepcopy(self.global_model.head.state_dict())
        self.client_heads = [copy.deepcopy(initial_head)
                             for _ in range(config.population_clients)]
        self.criterion = nn.CrossEntropyLoss()
        self.rng = np.random.default_rng(config.seed)
        self.robust_aggregator = None
        self.attack = None
        if config.byzantine_per_round:
            self.robust_aggregator = RobustAggregator(
                config.aggregator, config.byzantine_per_round)
            self.attack = ByzantineAttack(
                config.attack, config.byzantine_per_round,
                tau=config.attack_tau,
                epsilon=config.mimic_client,
            )

    def _aggregate(self, deltas: List[torch.Tensor], weights: List[int]) -> torch.Tensor:
        if self.cfg.byzantine_per_round == 0:
            weight = torch.tensor(weights, dtype=deltas[0].dtype,
                                  device=deltas[0].device)
            weight /= weight.sum()
            return sum(w * delta for w, delta in zip(weight, deltas))
        malicious = self.attack(deltas)
        return self.robust_aggregator(deltas + malicious)

    def train_round(self, round_number: int) -> float:
        chosen = self.rng.choice(self.cfg.population_clients,
                                 self.cfg.honest_per_round, replace=False)
        if self.cfg.algorithm == "fedrep":
            shared_params = list(self.global_model.representation.parameters())
        else:
            shared_params = list(self.global_model.parameters())
        shared_before = _flat(shared_params)
        deltas, weights, losses = [], [], []

        for client in chosen.tolist():
            local = copy.deepcopy(self.global_model)
            if self.cfg.algorithm == "fedrep":
                local.head.load_state_dict(self.client_heads[client])
            optimizer = _optimizer(local, self.cfg.lr, self.cfg.momentum)
            generator = torch.Generator().manual_seed(
                self.cfg.seed * 1_000_003 + round_number * 10_007 + client)

            if self.cfg.algorithm == "fedrep":
                _set_trainable(local, representation=False, head=True)
                _local_epochs(local, self.train_sets[client], self.cfg.head_epochs,
                              optimizer, self.criterion, self.cfg.batch_size,
                              self.device, generator)
                self.client_heads[client] = copy.deepcopy(local.head.state_dict())
                _set_trainable(local, representation=True, head=False)
                loss = _local_epochs(
                    local, self.train_sets[client], self.cfg.representation_epochs,
                    optimizer, self.criterion, self.cfg.batch_size,
                    self.device, generator)
                local_shared = _flat(local.representation.parameters())
            else:
                _set_trainable(local, representation=True, head=True)
                loss = _local_epochs(
                    local, self.train_sets[client], self.cfg.representation_epochs,
                    optimizer, self.criterion, self.cfg.batch_size,
                    self.device, generator)
                local_shared = _flat(local.parameters())

            deltas.append(local_shared - shared_before)
            weights.append(len(self.train_sets[client]))
            losses.append(loss)

        _set_flat(shared_params, shared_before + self._aggregate(deltas, weights))
        return float(np.mean(losses))

    def evaluate_all_clients(self) -> Tuple[float, float]:
        accuracies, losses = [], []
        for client in range(self.cfg.population_clients):
            model = copy.deepcopy(self.global_model)
            if self.cfg.algorithm == "fedrep":
                model.head.load_state_dict(self.client_heads[client])
            accuracy, loss = _evaluate(model, self.test_sets[client],
                                       self.cfg.batch_size, self.criterion,
                                       self.device)
            accuracies.append(accuracy)
            losses.append(loss)
        # Collins et al. use an unweighted mean of local client accuracies.
        return float(np.mean(accuracies)), float(np.mean(losses))

    def run(self) -> dict:
        history, train_history = [], []
        for round_number in range(1, self.cfg.rounds + 1):
            train_loss = self.train_round(round_number)
            train_history.append({"round": round_number, "loss": train_loss,
                                  "train_loss": train_loss,
                                  "learning_rate": self.cfg.lr})
            should_evaluate = (
                round_number % self.cfg.eval_every == 0 or
                round_number > self.cfg.rounds - 10
            )
            if should_evaluate:
                accuracy, test_loss = self.evaluate_all_clients()
                history.append({
                    "round": round_number,
                    "train_loss": train_loss,
                    "accuracy": accuracy,
                    "test_acc": accuracy,
                    "test_loss": test_loss,
                    "metric_name": "accuracy",
                    "metric_value": accuracy,
                    "global_acc": None,
                    "global_loss": None,
                    "global_metric_value": None,
                    "learning_rate": self.cfg.lr,
                })
                print(f"round={round_number:3d} train_loss={train_loss:.4f} "
                      f"mean_local_accuracy={accuracy:.4f}", flush=True)

        final_ten = [record for record in history
                     if record["round"] > self.cfg.rounds - 10]
        if len(final_ten) != min(10, self.cfg.rounds):
            raise RuntimeError("final-ten-round evaluation records are incomplete")
        final_metric = float(np.mean([r["accuracy"] for r in final_ten]))
        return {
            "format_version": 1,
            "benchmark": "collins21_femnist_letters",
            "dataset": "femnist_collins",
            "population_clients": self.cfg.population_clients,
            "n_clients": (self.cfg.honest_per_round +
                          self.cfg.byzantine_per_round),
            "n_byzantine": self.cfg.byzantine_per_round,
            "alpha": None,
            "loss_type": "cross_entropy",
            "algorithm": ("fedrep_nonlinear" if self.cfg.algorithm == "fedrep"
                          else "baseline"),
            "aggregator": self.cfg.aggregator,
            "attack": (self.cfg.attack if self.cfg.byzantine_per_round else "None"),
            "seed": self.cfg.seed,
            "repr_dim": 64,
            "head_steps": self.cfg.head_epochs,
            "metric_name": "accuracy",
            "config": asdict(self.cfg),
            "train_history": train_history,
            "history": history,
            "final_metric": final_metric,
            "best_metric": max(r["accuracy"] for r in history),
            "reporting_rule": "unweighted client mean, averaged over final 10 rounds",
        }

    def checkpoint(self) -> dict:
        cpu_heads = [
            {key: value.detach().cpu() for key, value in state.items()}
            for state in self.client_heads
        ]
        return {
            "format_version": 1,
            "benchmark": "collins21_femnist_letters",
            "config": asdict(self.cfg),
            "global_model_state_dict": {
                key: value.detach().cpu()
                for key, value in self.global_model.state_dict().items()
            },
            "client_head_state_dicts": cpu_heads,
        }
