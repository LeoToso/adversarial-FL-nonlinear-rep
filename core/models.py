"""
models.py  —  Neural network architectures
============================================
Two-part structure for FedRep:
  backbone  :  shared across clients (aggregated by server)  → repr_dim vector
  head      :  client-specific linear classifier             → n_classes logits

For the baseline single-model FL, backbone+head are both aggregated.

Backbones
---------
  CNNBackbone   — 3-block CNN for CIFAR-10/100 and FEMNIST images
  MLPBackbone   — 2-layer MLP for text (Sent140)
  LinearBackbone — single linear projection (linear setting from the paper)
"""

import torch
import torch.nn as nn
from typing import Tuple


# ── CNN backbone (images) ────────────────────────────────────────────────────

class CNNBackbone(nn.Module):
    """
    3-block CNN with BatchNorm + AdaptiveAvgPool → MLP projector.
    Works for any (C, H, W) input; always outputs repr_dim-dim vector.
    """
    def __init__(self, in_channels: int, repr_dim: int = 256):
        super().__init__()
        self.repr_dim = repr_dim
        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, padding=1),          nn.GroupNorm(8, 64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # block 2
            nn.Conv2d(64, 128, 3, padding=1),  nn.GroupNorm(16, 128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.GroupNorm(32, 256), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Sequential(
            nn.Linear(256 * 16, 512), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(512, repr_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return self.proj(h)


# ── MLP backbone (text / tabular) ────────────────────────────────────────────

class MLPBackbone(nn.Module):
    def __init__(self, input_dim: int, repr_dim: int = 256, hidden: int = 512):
        super().__init__()
        self.repr_dim = repr_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(hidden, hidden),    nn.ReLU(True), nn.Dropout(0.2),
            nn.Linear(hidden, repr_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


# ── Linear backbone (warm-up / linear setting of the paper) ──────────────────

class LinearBackbone(nn.Module):
    """B ∈ R^{repr_dim × d}  (no bias, matches the paper's linear model)."""
    def __init__(self, input_dim: int, repr_dim: int):
        super().__init__()
        self.repr_dim = repr_dim
        self.proj     = nn.Linear(input_dim, repr_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


# ── Linear head (client-specific) ────────────────────────────────────────────

class LinearHead(nn.Module):
    """h_i ∈ R^{n_classes × repr_dim}."""
    def __init__(self, repr_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(repr_dim, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


# ── Combined model (used by the baseline) ────────────────────────────────────

class FedModel(nn.Module):
    """backbone + head joined — used both by baseline FL and FedRep clients."""
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head     = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ── Factory helpers ───────────────────────────────────────────────────────────

def build_backbone(dataset: str, repr_dim: int, linear: bool = False) -> nn.Module:
    from core.datasets import DATASET_META
    meta = DATASET_META[dataset]
    if dataset in ("cifar10", "cifar100", "femnist", "isic2019"):
        if linear:
            return LinearBackbone(meta["dim"], repr_dim)
        return CNNBackbone(meta["in_ch"], repr_dim)
    elif dataset in ("sent140", "heart_disease", "school"):
        if linear:
            return LinearBackbone(meta["dim"], repr_dim)
        return MLPBackbone(meta["dim"], repr_dim)
    raise ValueError(f"Unknown dataset: {dataset}")


def build_model(dataset: str, repr_dim: int, n_classes: int,
                linear: bool = False) -> FedModel:
    bb   = build_backbone(dataset, repr_dim, linear)
    head = LinearHead(repr_dim, n_classes)
    return FedModel(bb, head)


# ── Flat-vector parameter utilities ──────────────────────────────────────────

def get_flat(params) -> torch.Tensor:
    return torch.cat([p.detach().view(-1) for p in params])


def set_flat(params, flat: torch.Tensor):
    offset = 0
    for p in params:
        n = p.numel()
        p.data.copy_(flat[offset: offset + n].view(p.shape))
        offset += n


def get_flat_grad(params) -> torch.Tensor:
    grads = []
    for p in params:
        g = p.grad
        grads.append(g.detach().view(-1) if g is not None
                     else torch.zeros(p.numel(), device=p.device))
    return torch.cat(grads)
