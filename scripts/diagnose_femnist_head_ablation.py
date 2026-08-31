"""Three training-only head-fitting cases; production FedRep is unchanged."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.utils.data import DataLoader
from core.datasets import get_loaders
from core.fed_fedrep import FedRep
from core.models import get_flat, get_flat_grad
from experiments.run_experiment import set_seed
from scripts.diagnose_femnist import save_json
from scripts.diagnose_femnist_optimization import features, score_report


def next_batch(iterator, loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def diagnostic_round(trainer, loaders, optimizer_name, dropout, seed, optimizers=None):
    """Same minibatches and backbone RNG across cases at each round/client.

    Optimizers reset each round unless a per-client optimizer list is supplied.
    The original CLI uses equal learning rates; other diagnostics may set
    trainer.lr_head explicitly and supply persistent optimizer objects.
    """
    records, momenta = [], []
    devices = [trainer.device.index if trainer.device.index is not None
               else torch.cuda.current_device()] if trainer.device.type == "cuda" else []
    for i, (model, source) in enumerate(zip(trainer.client_models, loaders)):
        phase_seed = seed + 100003 * trainer.round + 101 * i
        generator = torch.Generator().manual_seed(phase_seed)
        loader = DataLoader(source.dataset, batch_size=source.batch_size,
                            shuffle=True, generator=generator, num_workers=0)
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(phase_seed)
            # Full-writer deterministic training measurement before/after head fit.
            z, labels = features(model, source, trainer.device)
            with torch.no_grad():
                before = score_report(model.head(z), labels, trainer.criterion)
            model.train()
            if not dropout:
                for module in model.modules():
                    if isinstance(module, nn.modules.dropout._DropoutNd):
                        module.eval()
            model.zero_grad(set_to_none=True)
            model.backbone.requires_grad_(False)
            model.head.requires_grad_(True)
            optimizer = (optimizers[i] if optimizers is not None else
                         getattr(torch.optim, optimizer_name)(
                             model.head.parameters(), lr=trainer.lr_head))
            iterator = iter(loader)
            for _ in range(trainer.head_steps):
                (x, y), iterator = next_batch(iterator, loader)
                optimizer.zero_grad(set_to_none=True)
                loss = trainer.criterion(model(x.to(trainer.device)), y.to(trainer.device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite head loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.)
                optimizer.step()
            with torch.no_grad():
                model.head.eval()
                after = score_report(model.head(z), labels, trainer.criterion)
            # Restore dropout for the backbone phase in EVERY case.
            model.train()
            torch.manual_seed(phase_seed + 1)
            model.backbone.requires_grad_(True)
            model.head.requires_grad_(False)
            (x, y), iterator = next_batch(iterator, loader)
            model.zero_grad(set_to_none=True)
            loss = trainer.criterion(model(x.to(trainer.device)), y.to(trainer.device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite backbone loss")
            loss.backward()
            raw_norm = torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), 1.)
            gradient = get_flat_grad(model.backbone.parameters())
            model.zero_grad(set_to_none=True)
            model.requires_grad_(True)
            momentum = trainer.momentum * trainer.mom_buffers[i] + (1-trainer.momentum) * gradient
            trainer.mom_buffers[i] = momentum.detach()
            momenta.append(momentum.detach())
            records.append({"client": i, "before_head": before, "after_head": after,
                            "raw_backbone_gradient_norm": float(raw_norm)})
    old = get_flat(trainer.backbone.parameters()).clone()
    trainer._set_backbone_flat(old - trainer.lr_backbone * torch.stack(momenta).mean(0))
    trainer.round += 1
    return {"round": trainer.round, "clients": records,
            "backbone_update_norm": float((get_flat(trainer.backbone.parameters())-old).norm()),
            "mean_pre_head_loss": sum(c["before_head"]["loss"] for c in records)/len(records),
            "mean_post_head_loss": sum(c["after_head"]["loss"] for c in records)/len(records),
            "mean_post_head_train_accuracy": sum(c["after_head"]["accuracy"] for c in records)/len(records)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--output_dir", default="results/femnist_head_ablation_v4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=100)
    args = p.parse_args()
    if args.rounds < 1:
        p.error("rounds must be positive")
    root = Path(args.output_dir) / f"seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=False)
    print("Loading LEAF writers...", flush=True)
    train, _, _ = get_loaders("femnist", 10, 1., data_dir=args.data_dir,
                              batch_size=32, seed=args.seed, use_leaf=True)
    for name, optimizer, dropout in (("sgd_dropout", "SGD", True),
                                      ("sgd_no_dropout", "SGD", False),
                                      ("adam_no_dropout", "Adam", False)):
        set_seed(args.seed)
        trainer = FedRep("femnist", 10, 0, aggregator=None, attack=None,
                         repr_dim=128, head_steps=50, lr_head=.01,
                         lr_backbone=.1, momentum=.9, device=args.device,
                         loss_type="multiclass_ls")
        report = {"config": {"case": name, "optimizer": optimizer,
                  "head_dropout": dropout, "backbone_dropout": True,
                  "optimizer_reset_each_round": True, "lr_head": .01,
                  "lr_backbone": .1, "momentum": .9, "head_steps": 50,
                  "seed": args.seed, "rounds": args.rounds, "batch_size": 32,
                  "repr_dim": 128, "loss_type": "multiclass_ls", "n_honest": 10,
                  "n_byzantine": 0, "measurement_split": "training",
                  "measurement_timing": "before_server_backbone_update"}, "history": []}
        for _ in range(args.rounds):
            row = diagnostic_round(trainer, train, optimizer, dropout, args.seed)
            report["history"].append(row)
            save_json(root / name / "diagnostic.json", report)
            if row["round"] == 1 or row["round"] % 10 == 0 or row["round"] == args.rounds:
                print(f"{name}, round={row['round']}: head loss "
                      f"{row['mean_pre_head_loss']:.4f} -> {row['mean_post_head_loss']:.4f}, "
                      f"train_acc={row['mean_post_head_train_accuracy']:.4f}", flush=True)
        torch.save({"config": report["config"], "round": trainer.round,
                    "backbone_state_dict": trainer.backbone.state_dict(),
                    "client_head_state_dicts": [m.head.state_dict() for m in trainer.client_models]},
                   root / name / "final_model.pt")
        del trainer
    print(f"Saved three training-only cases to {root}", flush=True)


if __name__ == "__main__":
    main()
