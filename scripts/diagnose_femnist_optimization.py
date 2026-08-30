"""Training-only FedRep diagnostics; no test-based hyperparameter selection."""
import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from core.datasets import get_loaders
from core.fed_fedrep import FedRep
from core.models import get_flat
from experiments.run_experiment import set_seed
from scripts.diagnose_femnist import save_json


def score_report(scores, labels, objective):
    predictions = scores.argmax(1)
    counts = torch.bincount(predictions, minlength=scores.shape[1])
    majority = int(torch.bincount(labels, minlength=scores.shape[1]).argmax())
    return {"loss": float(objective(scores, labels)),
            "accuracy": float((predictions == labels).float().mean()),
            "prediction_histogram": counts.cpu().tolist(),
            "predicted_classes": int((counts > 0).sum()),
            "dominant_prediction_fraction": float(counts.max() / len(labels)),
            "majority_label": majority,
            "majority_accuracy": float((labels == majority).float().mean())}


@torch.no_grad()
def features(model, loader, device):
    model.eval()
    zs, ys = [], []
    # Deterministic order; never reuse the training iterator for this probe.
    for x, y in DataLoader(loader.dataset, batch_size=64, shuffle=False):
        zs.append(model.backbone(x.to(device)))
        ys.append(y.to(device))
    return torch.cat(zs), torch.cat(ys)


def head_probe(model, loader, objective, device, steps):
    """Fit copies of the head on cached training features, without dropout."""
    z, y = features(model, loader, device)
    output = {"samples": len(y), "feature_std_mean": float(z.std(0, unbiased=False).mean()),
              "dropout_enabled": False, "batch": "full_training_writer", "fits": {}}
    for name, lr in (("SGD", 0.01), ("Adam", 0.001)):
        head = copy.deepcopy(model.head)
        head.requires_grad_(True)
        optimizer = getattr(torch.optim, name)(head.parameters(), lr=lr)
        history = []
        for step in range(steps + 1):
            if step % 50 == 0 or step == steps:
                with torch.no_grad():
                    history.append({"step": step, **score_report(head(z), y, objective)})
            if step == steps:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = objective(head(z), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite frozen-head loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        output["fits"][name] = {"lr": lr, "clip_norm": 1.0, "history": history}
    return output


def run_case(train, loss_type, head_steps, rounds, probe_steps, device, seed, path):
    set_seed(seed)
    trainer = FedRep("femnist", len(train), 0,
                     aggregator=lambda vectors: torch.stack(vectors).mean(0),
                     attack=lambda vectors: [], repr_dim=128, head_steps=head_steps,
                     lr_head=0.01, lr_backbone=0.1, momentum=0.9,
                     device=device, loss_type=loss_type)
    gradient_sq = [0.0] * len(train)
    handles = []
    def hook(i):
        def record(g):
            gradient_sq[i] += float(g.detach().square().sum())
        return record
    for i, model in enumerate(trainer.client_models):
        for p in model.backbone.parameters():
            handles.append(p.register_hook(hook(i)))
    report = {"loss_type": loss_type, "head_steps": head_steps,
              "lr_head": 0.01, "lr_backbone": 0.1, "momentum": 0.9,
              "seed": seed, "n_honest": len(train), "n_byzantine": 0,
              "rounds": rounds, "metric_split": "training", "history": []}
    try:
        for rnd in range(1, rounds + 1):
            gradient_sq[:] = [0.0] * len(train)
            before = get_flat(trainer.backbone.parameters()).clone()
            train_stats = trainer.train_round(train)
            update_norm = float((get_flat(trainer.backbone.parameters()) - before).norm())
            row = {"round": rnd, "stochastic_head_loss": train_stats["loss"],
                   "backbone_raw_gradient_norms": [v ** 0.5 for v in gradient_sq],
                   "backbone_update_norm": update_norm,
                   "backbone_relative_update": update_norm / max(float(before.norm()), 1e-12)}
            if rnd == 1 or rnd % 10 == 0 or rnd == rounds:
                # DataLoader iteration consumes RNG even without shuffle.
                # Restore CPU/CUDA states so diagnostics don't alter training.
                devices = [torch.device(device).index or torch.cuda.current_device()] if str(device).startswith("cuda") else []
                with torch.random.fork_rng(devices=devices), torch.no_grad():
                    row["clients"] = []
                    for model, loader in zip(trainer.client_models, train):
                        z, y = features(model, loader, device)
                        row["clients"].append(score_report(model.head(z), y, trainer.criterion))
                row["mean_train_accuracy"] = sum(c["accuracy"] for c in row["clients"]) / len(train)
                print(f"{loss_type}, head_steps={head_steps}, round={rnd}: "
                      f"train_acc={row['mean_train_accuracy']:.4f}, update={update_norm:.3g}", flush=True)
            report["history"].append(row)
            save_json(path / "optimization.json", report)
        report["frozen_head_probes"] = []
        for i, (model, loader) in enumerate(zip(trainer.client_models, train)):
            print(f"Frozen-head probe: writer {i}, {loss_type}, head_steps={head_steps}", flush=True)
            report["frozen_head_probes"].append(head_probe(
                model, loader, trainer.criterion, device, probe_steps))
            save_json(path / "optimization.json", report)
        torch.save({"round": rounds, "config": {k: v for k, v in report.items()
                    if k not in {"history", "frozen_head_probes"}},
                    "backbone_state_dict": trainer.backbone.state_dict(),
                    "client_head_state_dicts": [m.head.state_dict() for m in trainer.client_models]},
                   path / "final_model.pt")
    finally:
        for handle in handles:
            handle.remove()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--output_dir", default="results/femnist_optimization_v3")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--probe_steps", type=int, default=200)
    args = p.parse_args()
    if args.rounds < 1 or args.probe_steps < 1:
        p.error("rounds and probe_steps must be positive")
    root = Path(args.output_dir) / f"seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=False)
    print("Loading LEAF training writers; initial JSON parsing may take minutes.", flush=True)
    train, _, _ = get_loaders("femnist", 10, 1.0, data_dir=args.data_dir,
                              batch_size=32, seed=args.seed, use_leaf=True)
    for loss in ("cross_entropy", "multiclass_ls"):
        for head_steps in (10, 50):
            run_case(train, loss, head_steps, args.rounds, args.probe_steps,
                     args.device, args.seed, root / loss / f"head_steps_{head_steps}")
    print(f"Saved training-only diagnostics to {root}. No test-based selection was performed.")


if __name__ == "__main__":
    main()
