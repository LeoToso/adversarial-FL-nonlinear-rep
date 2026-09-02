"""
aggregators.py  —  Robust aggregators and Byzantine attacks
=============================================================
Wraps ByzFL (https://github.com/LPD-EPFL/byzfl) with fallback
implementations when ByzFL is not installed.

(f,κ)-robust aggregators supported:
  TrMean          coordinate-wise trimmed mean  (κ = O(f/n), order-optimal)
  Krum            κ = O(1)  (suboptimal without NNM)
  GM              geometric median, κ = O(1)
  CWMed           coordinate-wise median, κ = O(1)
  Average         no robustness (honest-only baseline)
  NNM+<base>      NNM pre-aggregation → reduces κ to O(f/n) for any base

Byzantine attacks:
  SignFlipping, InnerProductManipulation (IPM), A Little Is Enough (ALIE),
  FallOfEmpires (FOE), Mimic, Zero, Random
"""

import torch
import numpy as np
from typing import List

# ── ByzFL import ─────────────────────────────────────────────────────────────

try:
    import byzfl as _bfl
    _BFL = True
except ImportError:
    _BFL = False
    print("[WARN] ByzFL not installed — using built-in fallback aggregators.")
    print("       Install: pip install byzfl")


# ── numpy / tensor conversions ────────────────────────────────────────────────

def _np(vecs: List[torch.Tensor]) -> np.ndarray:
    return np.stack([v.cpu().float().numpy() for v in vecs])


def _th(arr: np.ndarray, ref: torch.Tensor) -> torch.Tensor:
    return torch.tensor(arr, dtype=ref.dtype, device=ref.device)


# ── Fallback aggregators (no ByzFL required) ─────────────────────────────────

class _AvgFB:
    def __call__(self, vecs):
        return torch.stack(vecs).mean(0)

class _TrMeanFB:
    def __init__(self, f):  self.f = f
    def __call__(self, vecs):
        S = torch.stack(vecs)
        n = S.shape[0]
        if n - 2*self.f <= 0:
            return S.mean(0)
        S_sorted, _ = torch.sort(S, dim=0)
        return S_sorted[self.f: n - self.f].mean(0)

class _KrumFB:
    def __init__(self, f):  self.f = f
    def __call__(self, vecs):
        S = torch.stack(vecs); n = S.shape[0]; k = n - self.f - 2
        scores = []
        for i in range(n):
            d = torch.sum((S - S[i]) ** 2, dim=1)
            scores.append(torch.topk(d, k+1, largest=False).values[1:].sum())
        return S[torch.stack(scores).argmin()]

class _GMFb:
    def __init__(self, n_iter=80, eps=1e-6):  self.n_iter, self.eps = n_iter, eps
    def __call__(self, vecs):
        S = torch.stack(vecs).float(); m = S.mean(0)
        for _ in range(self.n_iter):
            d = torch.norm(S - m, dim=1, keepdim=True).clamp(min=self.eps)
            w = 1.0 / d; m = (w * S).sum(0) / w.sum()
        return m

class _CWMedFB:
    def __call__(self, vecs):
        return torch.stack(vecs).median(0).values

class _NNMFB:
    def __init__(self, f):  self.f = f
    def __call__(self, vecs):
        S = torch.stack(vecs); n = S.shape[0]; k = n - self.f
        out = []
        for i in range(n):
            d = torch.norm(S - S[i], dim=1)
            _, idx = torch.topk(d, k, largest=False)
            out.append(S[idx].mean(0))
        return out


# ── Public aggregator class ───────────────────────────────────────────────────

_SUPPORTED_AGG = [
    "Average", "TrMean", "Krum", "GM", "CWMed",
    "NNM+TrMean", "NNM+GM", "NNM+Krum", "NNM+CWMed",
]

class RobustAggregator:
    """
    Unified interface for robust aggregation rules.

    Parameters
    ----------
    name : str   one of _SUPPORTED_AGG
    f    : int   number of Byzantine workers (known upper bound)
    """

    def __init__(self, name: str, f: int):
        if name not in _SUPPORTED_AGG:
            raise ValueError(f"Unknown aggregator '{name}'. Options: {_SUPPORTED_AGG}")
        self.name = name
        self.f    = f
        self._nnm, self._base = self._build(name, f)

    def _build(self, name, f):
        use_nnm   = name.startswith("NNM+")
        base_name = name[4:] if use_nnm else name

        # --- NNM ---
        if use_nnm:
            nnm = _bfl.NNM(f=f) if _BFL else _NNMFB(f)
        else:
            nnm = None

        # --- base aggregator ---
        if _BFL:
            _map = {
                "Average": lambda: _bfl.Average(),
                "TrMean":  lambda: _bfl.TrMean(f=f),
                "Krum":    lambda: _bfl.Krum(f=f),
                "GM":      lambda: _bfl.GeometricMedian(),
                "CWMed":   lambda: _bfl.Median(),
            }
        else:
            _map = {
                "Average": lambda: _AvgFB(),
                "TrMean":  lambda: _TrMeanFB(f),
                "Krum":    lambda: _KrumFB(f),
                "GM":      lambda: _GMFb(),
                "CWMed":   lambda: _CWMedFB(),
            }
        base = _map[base_name]()
        return nnm, base

    def __call__(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        if not vectors:
            raise ValueError("Empty vector list.")

        # ByzFL accepts torch tensors directly.  Keeping the stacked client
        # matrix on its original device avoids copying tens of millions of
        # gradient coordinates GPU -> NumPy -> GPU every round.
        stacked = torch.stack(vectors)

        if self._nnm is not None:
            if _BFL:
                stacked = self._nnm(stacked)
            else:
                vectors = self._nnm(vectors)              # returns list of tensors
                stacked = torch.stack(vectors)

        if _BFL:
            return self._base(stacked)
        else:
            return self._base(vectors)


# ── Byzantine attacks ─────────────────────────────────────────────────────────

_SUPPORTED_ATK = [
    "SignFlipping", "InnerProductManipulation", "ALIE",
    "FallOfEmpires", "Mimic", "Zero", "Random",
]

class ByzantineAttack:
    """
    Returns f Byzantine gradient vectors given honest gradients.

    Parameters
    ----------
    name : str   attack name from _SUPPORTED_ATK
    f    : int   number of Byzantine workers
    **kw       : attack-specific parameters (e.g. tau=1.5, epsilon=0)
    """

    def __init__(self, name: str, f: int, **kw):
        if name not in _SUPPORTED_ATK:
            raise ValueError(f"Unknown attack '{name}'. Options: {_SUPPORTED_ATK}")
        self.name = name
        self.f    = f
        self.kw   = kw
        self._atk = self._build(name, f, **kw) if _BFL else None

    def _build(self, name, f, **kw):
        tau = kw.get("tau", 1.5)
        epsilon = kw.get("epsilon", 0)
        _map = {
            "SignFlipping":             lambda: _bfl.SignFlipping(),
            "InnerProductManipulation": lambda: _bfl.InnerProductManipulation(tau=tau),
            "ALIE":                     lambda: _bfl.ALittleIsEnough(tau=tau),
            "FallOfEmpires":            lambda: _bfl.FallOfEmpires(f=f, tau=tau),
            "Mimic":                    lambda: _bfl.Mimic(epsilon=epsilon),
            "Zero":                     lambda: None,
            "Random":                   lambda: None,
        }
        return _map[name]()

    def __call__(self, honest: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.f == 0:
            return []

        if _BFL and self._atk is not None:
            # ByzFL preserves torch tensors and their device, so the attack is
            # computed on GPU alongside training and robust aggregation.
            byz = self._atk(torch.stack(honest))      # single attack vector
            return [byz.clone() for _ in range(self.f)]

        # ── fallback ──
        S    = torch.stack(honest).float()
        mean = S.mean(0)
        tau  = self.kw.get("tau", 1.5)

        if self.name == "SignFlipping":
            byz = -mean
        elif self.name == "ALIE":
            # ByzFL and the ALIE definition use the population standard
            # deviation (denominator n), not PyTorch's unbiased estimator.
            byz = mean + tau * S.std(dim=0, unbiased=False)
        elif self.name in ("InnerProductManipulation", "FallOfEmpires"):
            byz = -tau * mean / (mean.norm() + 1e-8) * S.norm(dim=1).max()
        elif self.name == "Mimic":
            epsilon = self.kw.get("epsilon", 0)
            if not isinstance(epsilon, int) or not 0 <= epsilon < len(honest):
                raise ValueError(
                    "Mimic epsilon must be the zero-based index of an honest "
                    f"client in [0, {len(honest) - 1}]"
                )
            byz = S[epsilon].clone()
        elif self.name == "Zero":
            byz = torch.zeros_like(mean)
        else:  # Random
            byz = torch.randn_like(mean) * S.std()

        return [byz.clone() for _ in range(self.f)]
