"""
metrics.py  —  Result tracking and metric computation
======================================================
"""

import json
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class RoundRecord:
    round:     int
    train_loss: float
    test_acc:  Optional[float]
    test_loss: float
    wall_time: float
    global_acc: Optional[float] = None
    metric_name: str = "accuracy"
    metric_value: float = 0.0
    global_loss: Optional[float] = None
    global_metric_value: Optional[float] = None
    learning_rate: Optional[float] = None


@dataclass
class ExperimentResult:
    # Config
    dataset:     str
    n_clients:   int
    n_byzantine: int
    alpha:       float        # Dirichlet heterogeneity
    aggregator:  str
    attack:      str
    algorithm:   str          # "baseline" | "fedrep_linear" | "fedrep_nonlinear"
    repr_dim:    int
    head_steps:  int
    seed:        int
    loss_type:   str = "cross_entropy"
    task:        str = "classification"
    het_score:   float = 0.0

    # Per-round history
    history: List[RoundRecord] = field(default_factory=list)
    train_history: List[Dict] = field(default_factory=list)

    # Summary (filled after training)
    best_acc:    float = 0.0
    final_acc:   float = 0.0
    auc_acc:     float = 0.0   # area under accuracy curve (normalised)
    best_metric: float = 0.0
    final_metric: float = 0.0

    def add(self, round_rec: RoundRecord):
        self.history.append(round_rec)
        values = [r.metric_value for r in self.history]
        self.best_metric = (max(values) if round_rec.metric_name == "accuracy"
                            else min(values))
        self.final_metric = values[-1]
        accs = [r.test_acc for r in self.history if r.test_acc is not None]
        if accs:
            self.best_acc = max(accs)
            self.final_acc = accs[-1]
            trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            self.auc_acc = float(trap(accs) / max(len(accs) - 1, 1))

    def add_train(
        self,
        round_number: int,
        objective: float,
        wall_time: float,
        learning_rate: Optional[float] = None,
    ):
        self.train_history.append({
            "round": int(round_number),
            "train_loss": float(objective),
            "wall_time": float(wall_time),
            "learning_rate": (None if learning_rate is None
                              else float(learning_rate)),
        })

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["history"] = [asdict(r) for r in self.history]
        return d

    def to_row(self) -> Dict:
        """Flat row suitable for pandas DataFrame."""
        return {
            "dataset":     self.dataset,
            "n_clients":   self.n_clients,
            "n_byzantine": self.n_byzantine,
            "byz_fraction": self.n_byzantine / self.n_clients,
            "alpha":       self.alpha,
            "aggregator":  self.aggregator,
            "attack":      self.attack,
            "algorithm":   self.algorithm,
            "repr_dim":    self.repr_dim,
            "head_steps":  self.head_steps,
            "seed":        self.seed,
            "best_acc":    self.best_acc,
            "final_acc":   self.final_acc,
            "auc_acc":     self.auc_acc,
        }


class Timer:
    def __init__(self):
        self._start = time.time()

    def elapsed(self) -> float:
        return time.time() - self._start


def save_results(results: List[ExperimentResult], path: str):
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)


def load_results(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)
