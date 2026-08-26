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
    test_acc:  float
    test_loss: float
    wall_time: float
    global_acc: float = 0.0   


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

    # Per-round history
    history: List[RoundRecord] = field(default_factory=list)

    # Summary (filled after training)
    best_acc:    float = 0.0
    final_acc:   float = 0.0
    auc_acc:     float = 0.0   # area under accuracy curve (normalised)

    def add(self, round_rec: RoundRecord):
        self.history.append(round_rec)
        accs = [r.test_acc for r in self.history]
        self.best_acc  = max(accs)
        self.final_acc = accs[-1]
        self.auc_acc   = float(np.trapezoid(accs) / max(len(accs) - 1, 1)) if hasattr(np, 'trapezoid') else float(np.trapz(accs) / max(len(accs) - 1, 1))

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
