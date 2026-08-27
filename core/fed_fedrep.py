"""
fed_fedrep.py  —  Adversarial Federated Representation Learning
================================================================
Implements the algorithm from:
  "Heterogeneity is Not Fundamental in Robust Federated Feature Learning"
  (Zhang, Toso, Anderson, Matni)

which extends FedRep (Collins et al., ICML 2021) to the Byzantine-resilient
setting via alternating minimisation + robust aggregation on the representation
(backbone) coordinate only.

Algorithm per round t  (matches Algorithm in the paper)
─────────────────────────────────────────────────────────
1. Server broadcasts shared backbone B_{t-1} to every client.

2. Each honest client i  [HEAD STEP — local, not aggregated]:
     For τ_h local SGD steps on the HEAD only (backbone frozen):
       h_i ← h_i - η_h · ∇_{h_i} f_i(h_i ; B_{t-1})

3. Each honest client i  [REPRESENTATION STEP — one gradient step]:
     Compute stochastic gradient  g^(i)_B  w.r.t. backbone params,
     using the freshly updated local head h_i.
     Update local momentum:
       m^(i)_t = β·m^(i)_{t-1} + (1-β)·g^(i)_B

4. Byzantine clients send crafted vectors (attack on backbone gradients).

5. Server applies robust aggregation F (optionally with NNM):
     R_t = F(m^(1)_t, …, m^(n)_t)

6. Server backbone update (SHB style):
     B_t = B_{t-1} - γ · R_t

Key property (Theorem D.2 in paper):
  G²_B (representation gradient heterogeneity) → 0 as τ_h → ∞,
  eliminating the non-vanishing error floor that plagues single-model FL.
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

from core.models      import FedModel, build_model, get_flat, set_flat, get_flat_grad
from core.aggregators import RobustAggregator, ByzantineAttack
from core.datasets    import DATASET_META
from core.objectives  import TaskObjective, evaluate_model


class FedRep:
    """
    Adversarial federated representation learning.

    The server owns and aggregates only the backbone (shared representation).
    Each client keeps its own head, which is optimised locally for τ_head steps
    before the backbone gradient is computed.

    Parameters
    ----------
    dataset      : dataset name
    n_clients    : total clients (honest + Byzantine)
    n_byzantine  : Byzantine count  (f < n/2)
    aggregator   : RobustAggregator (applied to backbone gradients only)
    attack       : ByzantineAttack  (attacks backbone gradients)
    repr_dim     : shared backbone output dimension
    lr_backbone  : server learning rate γ for backbone update
    lr_head      : client learning rate η_h for head update
    head_steps   : τ_h  — local head optimisation steps per round
    momentum     : SHB β for backbone momentum
    linear       : use linear backbone (paper's linear setting)
    device       : torch device
    """

    def __init__(
        self,
        dataset:      str,
        n_clients:    int,
        n_byzantine:  int,
        aggregator:   RobustAggregator,
        attack:       ByzantineAttack,
        repr_dim:     int   = 256,
        lr_backbone:  float = 0.1,
        lr_head:      float = 0.01,
        head_steps:   int   = 10,
        momentum:     float = 0.9,
        linear:       bool  = False,
        device:       str   = "cpu",
        loss_type:    str   = "cross_entropy",
    ):
        assert n_byzantine < n_clients / 2
        meta = DATASET_META[dataset]

        self.n_clients   = n_clients
        self.n_byzantine = n_byzantine
        self.n_honest    = n_clients - n_byzantine
        self.aggregator  = aggregator
        self.attack      = attack
        self.lr_backbone = lr_backbone
        self.lr_head     = lr_head
        self.head_steps  = head_steps
        self.momentum    = momentum
        self.device      = torch.device(device)
        self.n_classes   = meta["n_classes"]
        self.task        = meta.get("task", "classification")
        self.client_loaders = None  # set externally after construction
        self.client_test_loaders = None  # set externally

        # ── server backbone (shared) ────────────────────────────────────────
        # We build a full model but the server only stores / aggregates backbone.
        _ref = build_model(dataset, repr_dim, meta["n_classes"], linear)
        self.backbone = _ref.backbone.to(self.device)

        # ── per-client: each honest client has its own full model ────────────
        # (backbone initialised from server copy; head initialised randomly)
        self.client_models: List[FedModel] = [
            build_model(dataset, repr_dim, meta["n_classes"], linear).to(self.device)
            for _ in range(self.n_honest)
        ]
        # Sync all client backbones to server backbone at init
        self._broadcast_backbone()

        # ── SHB momentum buffers for backbone coordinate ─────────────────────
        n_bb = sum(p.numel() for p in self.backbone.parameters())
        self.mom_buffers = [
            torch.zeros(n_bb, device=self.device)
            for _ in range(self.n_honest)
        ]

        self.criterion = TaskObjective(self.task, loss_type, meta["n_classes"])
        self.round = 0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _broadcast_backbone(self):
        """Copy server backbone weights to all client models."""
        bb_flat = get_flat(self.backbone.parameters())
        for cm in self.client_models:
            set_flat(cm.backbone.parameters(), bb_flat.clone())

    def _backbone_flat(self) -> torch.Tensor:
        return get_flat(self.backbone.parameters())

    def _set_backbone_flat(self, flat: torch.Tensor):
        set_flat(self.backbone.parameters(), flat)
        self._broadcast_backbone()

    # ── one round ────────────────────────────────────────────────────────────

    def train_round(self, client_loaders: List[DataLoader]) -> Dict[str, float]:
        """
        Parameters
        ----------
        client_loaders : DataLoaders for honest clients  (len == n_honest)

        Returns
        -------
        dict with 'loss' (mean backbone gradient-momentum norm²)
        """
        honest_momenta: List[torch.Tensor] = []
        ce_losses: List[float] = []

        for i, loader in enumerate(client_loaders[: self.n_honest]):
            model = self.client_models[i]
            model.train()

            # ── Step 2: local head update (τ_h steps, backbone frozen) ───────
            for p in model.backbone.parameters():
                p.requires_grad_(False)
            for p in model.head.parameters():
                p.requires_grad_(True)

            head_opt = torch.optim.SGD(model.head.parameters(), lr=self.lr_head)

            # Materialise a batch iterator that wraps around
            loader_iter = iter(loader)
            for _ in range(self.head_steps):
                try:
                    x, y = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    x, y = next(loader_iter)

                x, y = x.to(self.device), y.to(self.device)
                head_opt.zero_grad()
                loss = self.criterion(model(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                ce_losses.append(loss.item()) 
                head_opt.step()

            # ── Step 3: backbone gradient (head frozen) ───────────────────────
            for p in model.backbone.parameters():
                p.requires_grad_(True)
            for p in model.head.parameters():
                p.requires_grad_(False)

            # one mini-batch for backbone gradient
            try:
                x, y = next(loader_iter)
            except StopIteration:
                x, y = next(iter(loader))
            x, y = x.to(self.device), y.to(self.device)

            model.zero_grad()
            loss = self.criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            g_bb = get_flat_grad(model.backbone.parameters())

            # restore all params to trainable
            for p in model.parameters():
                p.requires_grad_(True)

            # SHB momentum update
            m_prev = self.mom_buffers[i]
            m_new  = self.momentum * m_prev + (1.0 - self.momentum) * g_bb
            self.mom_buffers[i] = m_new.detach()
            honest_momenta.append(m_new.detach())

        # ── Step 4-5: Byzantine attack + robust aggregation ───────────────────
        byz_vectors = self.attack(honest_momenta)
        all_vectors = honest_momenta + byz_vectors
        agg_g = self.aggregator(all_vectors)

        # ── Step 6: server backbone update ────────────────────────────────────
        new_bb = self._backbone_flat() - self.lr_backbone * agg_g
        self._set_backbone_flat(new_bb)

        self.round += 1
        return {"loss": float(sum(ce_losses) / len(ce_losses))}

    # ── evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, test_loader, eval_client_idx=None):
        client_accs, client_losses = [], []
        for i, cm in enumerate(self.client_models):
            # Evaluation must be read-only: local heads have already been
            # trained during every communication round.  Updating them here
            # would make the optimization trajectory depend on eval_every.
            set_flat(cm.backbone.parameters(), self._backbone_flat().clone())
            cm.eval()
            metrics = evaluate_model(cm, self.client_test_loaders[i],
                                     self.criterion, self.device)
            client_accs.append(metrics["metric_value"])
            client_losses.append(metrics["test_loss"])
        value = float(sum(client_accs)/len(client_accs))
        out = {"metric_name": "accuracy" if self.task == "classification" else "mse",
               "metric_value": value,
               "test_loss": float(sum(client_losses)/len(client_losses))}
        out["test_acc" if self.task == "classification" else "test_mse"] = value
        return out
    
    def evaluate_global(
        self,
        test_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Global evaluation: freeze the backbone, fit a fresh head on the union
        of honest training sets, and evaluate it on held-out global test data.
        This measures backbone quality without fitting on the test set.
        """
        probe = copy.deepcopy(self.client_models[0])
        set_flat(probe.backbone.parameters(), self._backbone_flat().clone())

        # Freeze backbone, reset head
        for p in probe.backbone.parameters():
            p.requires_grad_(False)
        for p in probe.head.parameters():
            torch.nn.init.xavier_uniform_(p) if p.dim() > 1 else torch.nn.init.zeros_(p)
            p.requires_grad_(True)

        # Train a fresh global head on the union of honest *training* sets.
        probe.train()
        opt = torch.optim.SGD(probe.head.parameters(), lr=0.1, momentum=0.9)
        for _ in range(3):
            for train_loader in self.client_loaders:
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    opt.zero_grad()
                    loss = self.criterion(probe(x), y)
                    loss.backward()
                    opt.step()

        # Evaluate only on the held-out global test loader.
        return evaluate_model(probe, test_loader, self.criterion, self.device)

    @torch.no_grad()
    def evaluate_per_client(
        self, client_loaders: List[DataLoader]
    ) -> List[Dict[str, float]]:
        """Personalised evaluation: each client uses its own head."""
        results = []
        for i, loader in enumerate(client_loaders[: self.n_honest]):
            model = self.client_models[i]
            model.eval()
            correct = total = 0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                correct += (model(x).argmax(1) == y).sum().item()
                total   += y.size(0)
            results.append({"client_acc": correct / total})
        return results

    def get_backbone(self):
        return copy.deepcopy(self.backbone)
