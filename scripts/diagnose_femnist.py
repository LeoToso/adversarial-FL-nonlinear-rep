#!/usr/bin/env python3
"""Check real LEAF data, fixed-batch fitting, and clean (f=0) FL/FedRep.

These are diagnostics, not paper results. The fixed-batch Adam fit tests basic
model/data learnability, not the federated optimizer. No test data are used to fit.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from core.datasets import DATASET_META, get_loaders
from core.models import build_model
from core.objectives import TaskObjective
from experiments.run_experiment import run_experiment, set_seed


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
    os.replace(temporary, path)


def fixed_batch_fit(x, y, loss_type, steps, device, seed):
    set_seed(seed)
    model = build_model("femnist", 128, 62).to(device)
    objective = TaskObjective("classification", loss_type, 62)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    x, y = x.to(device), y.to(device)
    history = []
    for step in range(steps + 1):
        if step:
            # Disable dropout for this deterministic memorization diagnostic.
            # eval() does not disable gradients; the paper trainers are unchanged.
            model.eval()
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite fixed-batch training loss")
            loss.backward()
            optimizer.step()
        if step % 25 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                scores = model(x)
                record = {"step": step, "loss": float(objective(scores, y)),
                          "accuracy": float((scores.argmax(1) == y).float().mean())}
            history.append(record)
            print(f"  fixed batch / {loss_type}: {record}", flush=True)
    return {"loss_type": loss_type, "seed": seed, "optimizer": "Adam",
            "lr": 0.0001, "dropout_enabled": False, "steps": steps, "samples": len(y),
            "distinct_labels": int(y.unique().numel()), "history": history,
            "passed_95pct_training_accuracy": history[-1]["accuracy"] >= 0.95}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="results/femnist_diagnostics_v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--mode", choices=["all", "data", "overfit", "clean"], default="all")
    parser.add_argument("--losses", nargs="+", choices=["cross_entropy", "multiclass_ls"],
                        default=["cross_entropy", "multiclass_ls"])
    args = parser.parse_args()
    if args.steps < 1 or args.rounds < 1:
        parser.error("steps and rounds must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    root = Path(args.output_dir) / f"seed_{args.seed}"
    if root.exists():
        raise FileExistsError(f"{root} already exists; use a fresh --output_dir")

    set_seed(args.seed)
    print("Loading and validating LEAF JSON (the first load may take minutes)...", flush=True)
    train, test, _ = get_loaders("femnist", 10, 1.0, data_dir=args.data_dir,
                                batch_size=32, seed=args.seed, use_leaf=True)
    assert DATASET_META["femnist"]["n_classes"] == 62
    clients = []
    for i, (tr, te) in enumerate(zip(train, test)):
        # LEAF stores normalized tensors directly in these per-writer datasets.
        raw = tr.dataset.x * 0.3081 + 0.1307
        labels = tr.dataset.y
        counts = torch.bincount(labels, minlength=62)
        majority = int(counts.argmax())
        clients.append({"client_index": i, "train_samples": len(tr.dataset),
                        "test_samples": len(te.dataset),
                        "train_class_count": int((counts > 0).sum()),
                        "raw_pixel_min": float(raw.min()),
                        "raw_pixel_max": float(raw.max()),
                        "mean_within_image_pixel_std": float(raw.flatten(1).std(1).mean()),
                        "train_majority_label": majority,
                        "majority_predictor_test_accuracy": float(
                            (te.dataset.y == majority).float().mean())})
    report = {"dataset": "femnist", "use_leaf": True, "n_classes": 62,
              "n_honest": 10, "seed": args.seed, "clients": clients}
    save_json(root / "data_check.json", report)
    print(json.dumps(report, indent=2), flush=True)

    if args.mode in {"all", "overfit"}:
        x, y = next(iter(train[0]))
        for loss_type in args.losses:
            diagnostic = fixed_batch_fit(x[:16], y[:16], loss_type,
                                         args.steps, args.device, args.seed)
            save_json(root / f"fixed_batch_{loss_type}.json", diagnostic)
            if not diagnostic["passed_95pct_training_accuracy"]:
                raise RuntimeError(
                    f"Fixed-batch {loss_type} fit did not reach 95% training accuracy. "
                    "Inspect its saved history before launching more experiments."
                )

    if args.mode in {"all", "clean"}:
        for loss_type in args.losses:
            for algorithm in ("baseline", "fedrep_nonlinear"):
                path = root / "clean" / loss_type / algorithm / "result.json"
                result = run_experiment(
                    dataset="femnist", alpha=1.0, n_clients=10, n_byzantine=0,
                    algorithm=algorithm, aggregator="Average", attack="SignFlipping",
                    loss_type=loss_type, repr_dim=128, head_steps=10,
                    lr=0.1, lr_head=0.01, lr_schedule="constant", momentum=0.9,
                    rounds=args.rounds, batch_size=32, seed=args.seed,
                    data_dir=args.data_dir, device=args.device, use_leaf=True,
                    eval_every=10, checkpoint_path=str(path.with_suffix(".pt")))
                save_json(path, result.to_dict())
        print("Clean diagnostics completed; f=0 means no adversarial vectors were sent.")
    print(f"Saved diagnostics in {root}", flush=True)


if __name__ == "__main__":
    main()
