"""Clean SGD/reset-Adam/persistent-Adam comparison; not a paper-results grid."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from core.datasets import get_loaders
from core.fed_fedrep import FedRep
from experiments.run_experiment import set_seed
from scripts.diagnose_femnist import save_json
from scripts.diagnose_femnist_head_ablation import diagnostic_round
from scripts.diagnose_femnist_optimization import features, score_report


@torch.no_grad()
def evaluate_splits(trainer, train, heldout):
    """Read-only metrics after server update; preserve RNG and module modes."""
    devices = [trainer.device.index if trainer.device.index is not None
               else torch.cuda.current_device()] if trainer.device.type == "cuda" else []
    modes = [(m, m.training) for model in trainer.client_models for m in model.modules()]
    report = {}
    try:
        with torch.random.fork_rng(devices=devices):
            for split, loaders in (("train", train), ("heldout", heldout)):
                clients = []
                for model, loader in zip(trainer.client_models, loaders):
                    z, y = features(model, loader, trainer.device)
                    clients.append(score_report(model.head(z), y, trainer.criterion))
                report[split] = {"clients": clients,
                    "accuracy": sum(c["accuracy"] for c in clients)/len(clients),
                    "loss": sum(c["loss"] for c in clients)/len(clients)}
    finally:
        for module, mode in modes:
            module.training = mode
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--output_dir", default="results/femnist_optimizer_state_v5")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=100)
    args = p.parse_args()
    if args.rounds < 1:
        p.error("rounds must be positive")
    root = Path(args.output_dir) / f"seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=False)
    print("Loading LEAF writers...", flush=True)
    train, heldout, _ = get_loaders("femnist", 10, 1., data_dir=args.data_dir,
                                   batch_size=32, seed=args.seed, use_leaf=True)
    for name, opt, lr, persistent in (("sgd", "SGD", .01, False),
                                    ("adam_reset", "Adam", .001, False),
                                    ("adam_persistent", "Adam", .001, True)):
        set_seed(args.seed)
        trainer = FedRep("femnist", 10, 0, None, None, repr_dim=128,
                         head_steps=50, lr_head=lr, lr_backbone=.1, momentum=.9,
                         device=args.device, loss_type="multiclass_ls")
        optimizers = ([torch.optim.Adam(m.head.parameters(), lr=lr)
                       for m in trainer.client_models] if persistent else None)
        config = {"case": name, "optimizer": opt, "lr_head": lr,
                  "persistent_head_optimizer": persistent, "dropout": True,
                  "lr_backbone": .1, "momentum": .9, "head_steps": 50,
                  "repr_dim": 128, "batch_size": 32, "seed": args.seed,
                  "rounds": args.rounds, "n_honest": 10, "n_byzantine": 0,
                  "loss_type": "multiclass_ls", "metric_aggregation": "equal_client_mean",
                  "heldout_source": "LEAF test split; diagnostic, not validation-selected",
                  "evaluation_timing": "after_server_update"}
        report = {"config": config, "history": [], "evaluation": []}
        for _ in range(args.rounds):
            row = diagnostic_round(trainer, train, opt, True, args.seed, optimizers)
            report["history"].append(row)
            if trainer.round % 10 == 0 or trainer.round == args.rounds:
                metrics = evaluate_splits(trainer, train, heldout)
                report["evaluation"].append({"round": trainer.round, **metrics})
                print(f"{name}, round={trainer.round}: train_acc={metrics['train']['accuracy']:.4f}, "
                      f"heldout_acc={metrics['heldout']['accuracy']:.4f}, "
                      f"heldout_loss={metrics['heldout']['loss']:.4f}", flush=True)
            save_json(root / name / "diagnostic.json", report)
        torch.save({"config": config, "round": trainer.round,
                    "backbone_state_dict": trainer.backbone.state_dict(),
                    "client_head_state_dicts": [m.head.state_dict() for m in trainer.client_models],
                    "head_optimizer_state_dicts": [o.state_dict() for o in optimizers] if optimizers else None},
                   root / name / "final_model.pt")
        del optimizers, trainer
    print(f"Saved three cases to {root}. No automatic selection or early stopping.", flush=True)


if __name__ == "__main__":
    main()
