"""
fed_baseline.py  —  Adversarial Federated Learning (single global model)
=========================================================================
Implements D-SHB (distributed stochastic heavy-ball) with robust
aggregation, i.e. Algorithm 3 from Allouah et al. (AISTATS 2023).

All parameters (backbone + head) form a single flat vector that is
broadcast to clients, differentiated locally, and robustly aggregated
server-side.  This is the BASELINE against which FedRep is compared.

Algorithm per round t
─────────────────────
1. Server broadcasts global model θ_{t-1} to every client.
2. Each honest client i:
     a. Computes stochastic gradient  g^(i)_t  on one mini-batch.
     b. Updates local momentum  m^(i)_t = β·m^(i)_{t-1} + (1-β)·g^(i)_t
     c. Sends m^(i)_t to server.
3. Byzantine clients send crafted vectors via the chosen attack.
4. Server applies robust aggregation F (possibly with NNM pre-agg).
5. Server update:  θ_t = θ_{t-1} - γ · F(m^(1),...,m^(n))
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

from core.models      import FedModel, build_model, get_flat, set_flat, get_flat_grad
from core.aggregators import RobustAggregator, ByzantineAttack
from core.datasets    import DATASET_META


class FedBaseline:
    """
    Adversarial FL with a single shared global model (backbone + head).

    Parameters
    ----------
    dataset       : dataset name
    n_clients     : total number of clients (honest + Byzantine)
    n_byzantine   : number of Byzantine clients  (f < n/2)
    aggregator    : RobustAggregator instance
    attack        : ByzantineAttack instance
    repr_dim      : shared representation dimension
    lr            : server learning rate γ
    momentum      : SHB momentum coefficient β
    linear        : use linear backbone (paper's linear setting)
    device        : torch device
    """

    def __init__(
        self,
        dataset:     str,
        n_clients:   int,
        n_byzantine: int,
        aggregator:  RobustAggregator,
        attack:      ByzantineAttack,
        repr_dim:    int    = 256,
        lr:          float  = 0.1,
        momentum:    float  = 0.9,
        linear:      bool   = False,
        device:      str    = "cpu",
    ):
        assert n_byzantine < n_clients / 2, "Need f < n/2 for Byzantine resilience."
        meta = DATASET_META[dataset]

        self.n_clients   = n_clients
        self.n_byzantine = n_byzantine
        self.n_honest    = n_clients - n_byzantine
        self.aggregator  = aggregator
        self.attack      = attack
        self.lr          = lr
        self.momentum    = momentum
        self.device      = torch.device(device)
        self.client_test_loaders = None  # set externally

        # Global model (server copy)
        self.model = build_model(
            dataset, repr_dim, meta["n_classes"], linear
        ).to(self.device)

        # Per-honest-client local momentum buffers  (flat vectors)
        n_params = sum(p.numel() for p in self.model.parameters())
        self.mom_buffers = [
            torch.zeros(n_params, device=self.device)
            for _ in range(self.n_honest)
        ]

        self.criterion = nn.CrossEntropyLoss()
        self.round = 0

    # ── single training round ────────────────────────────────────────────────

    def train_round(
        self,
        client_loaders: List[DataLoader],
    ) -> Dict[str, float]:
        """
        Run one communication round.

        Parameters
        ----------
        client_loaders : training loaders for honest clients only
                         (len == self.n_honest)

        Returns
        -------
        dict with 'loss' (mean honest gradient norm²)
        """
        self.model.train()
        global_flat = get_flat(self.model.parameters()).clone()

        honest_momenta: List[torch.Tensor] = []
        ce_losses: List[float] = []

        for i, loader in enumerate(client_loaders[: self.n_honest]):
            # Load one mini-batch
            x, y = next(iter(loader))
            x, y = x.to(self.device), y.to(self.device)

            # Compute gradient on current global model
            self.model.zero_grad()
            loss = self.criterion(self.model(x), y)
            loss.backward()
            ce_losses.append(loss.item())
            g = get_flat_grad(self.model.parameters())

            # Update local SHB momentum
            m_prev = self.mom_buffers[i]
            m_new  = self.momentum * m_prev + (1.0 - self.momentum) * g
            self.mom_buffers[i] = m_new.detach()
            honest_momenta.append(m_new.detach())

        # Byzantine attack
        byz_vectors = self.attack(honest_momenta)

        # Robust aggregation
        all_vectors = honest_momenta + byz_vectors
        agg = self.aggregator(all_vectors)

        # Server model update
        new_flat = global_flat - self.lr * agg
        set_flat(self.model.parameters(), new_flat)
        self.round += 1

        mean_gnorm = float(torch.stack([v.norm() for v in honest_momenta]).mean())
        return {"loss": float(sum(ce_losses) / len(ce_losses))}

    # ── evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, test_loader, client_loaders=None, head_steps=20, lr_head=0.01):
        self.model.eval()
        if self.client_test_loaders is None:
            # fallback to global eval
            correct = total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    correct += (self.model(x).argmax(1) == y).sum().item()
                    total += y.size(0)
            return {"test_acc": correct/total, "test_loss": 0.0}
        
        client_accs, client_losses = [], []
        for loader in self.client_test_loaders:
            correct = total = 0
            total_loss = 0.0
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    logits = self.model(x)
                    total_loss += self.criterion(logits, y).item() * y.size(0)
                    correct += (logits.argmax(1) == y).sum().item()
                    total += y.size(0)
            client_accs.append(correct / total)
            client_losses.append(total_loss / total)
        return {"test_acc": float(sum(client_accs)/len(client_accs)),
                "test_loss": float(sum(client_losses)/len(client_losses))}
    
    def evaluate_global(
        self,
        test_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Global evaluation using the shared model directly on test set.
        No fine-tuning — just the global model as-is.
        """
        self.model.eval()
        correct = total = 0
        total_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                total_loss += self.criterion(logits, y).item() * y.size(0)
                correct    += (logits.argmax(1) == y).sum().item()
                total      += y.size(0)
        return {
            "test_acc":  correct / total,
            "test_loss": total_loss / total,
        }

    def get_model(self) -> FedModel:
        return copy.deepcopy(self.model)
