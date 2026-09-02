#!/usr/bin/env python3
"""Plot five-seed learning curves for H=10,20,50 honest clients/round."""
import argparse, json, re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--dataset", required=True, choices=["cifar10", "femnist", "school"])
    p.add_argument("--output", required=True)
    return p.parse_args()


def identity(path, obj):
    text = str(path)
    h = (obj.get("honest_per_round") or
         obj.get("config", {}).get("honest_per_round") or
         obj.get("config", {}).get("active_honest_clients"))
    if h is None:
        m = re.search(r"honest_(10|20|50)", text); h = int(m.group(1)) if m else None
    method = obj.get("algorithm") or obj.get("config", {}).get("algorithm")
    if method == "baseline": method = "fedavg"
    agg = obj.get("aggregator") or obj.get("config", {}).get("aggregator")
    attack = obj.get("attack") or obj.get("config", {}).get("attack")
    return h, method, agg, attack


def series(obj, school):
    out = {}
    for rec in obj.get("history", []):
        value = rec.get("mean_local_accuracy")
        if value is None: value = rec.get("accuracy")
        if value is None: value = rec.get("metric_value")
        if value is None and school: value = rec.get("test_mse")
        if value is not None: out[int(rec["round"])] = float(value)
    return out


def main():
    a = args(); groups = defaultdict(list); school = a.dataset == "school"
    for path in Path(a.results_dir).rglob("seed_*.json"):
        with path.open() as f: obj = json.load(f)
        key = identity(path, obj)
        if key[0] in (10,20,50): groups[key].append(series(obj, school))
    panels = sorted({k[1:] for k in groups})
    if not panels: raise SystemExit("No matching result histories found")
    fig, axes = plt.subplots(2, max(1, (len(panels)+1)//2), figsize=(5.2*max(1,(len(panels)+1)//2), 8), squeeze=False)
    colors = {10:"#1f77b4",20:"#ff7f0e",50:"#2ca02c"}
    for ax, panel in zip(axes.flat, panels):
        for h in (10,20,50):
            runs = groups.get((h,)+panel, [])
            if not runs: continue
            rounds = sorted(set.intersection(*(set(x) for x in runs)))
            vals = np.array([[x[r] for r in rounds] for x in runs])
            mu, sd = vals.mean(0), vals.std(0, ddof=1) if len(runs)>1 else np.zeros(len(rounds))
            ax.plot(rounds, mu, label=f"H={h}", color=colors[h])
            ax.fill_between(rounds, mu-sd, mu+sd, color=colors[h], alpha=.18)
        ax.set_title(" / ".join(str(x) for x in panel)); ax.grid(alpha=.25); ax.legend()
        ax.set_xlabel("Communication round"); ax.set_ylabel("Test MSE (lower is better)" if school else "Mean local accuracy")
    for ax in axes.flat[len(panels):]: ax.remove()
    fig.suptitle(f"{a.dataset}: fixed f=5, varying honest clients per round")
    fig.tight_layout(); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")

if __name__ == "__main__": main()
