"""Provisional FEMNIST grid: unchanged baseline versus reset-Adam private heads."""
import argparse
import gc
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_femnist import save_json


def configurations(smoke=False):
    configs = []
    for loss in ("multiclass_ls", "cross_entropy"):
        for algorithm in ("fedrep_nonlinear", "baseline"):
            for agg in ("NNM+TrMean", "NNM+Krum"):
                for attack in ("SignFlipping", "InnerProductManipulation"):
                    for seed in ([42] if smoke else [42, 123, 456, 789, 1024]):
                        configs.append(dict(dataset="femnist", alpha=1., n_clients=15,
                            n_byzantine=5, loss_type=loss, algorithm=algorithm,
                            aggregator=agg, attack=attack, seed=seed, repr_dim=128,
                            head_steps=50 if algorithm == "fedrep_nonlinear" else 10,
                            lr=.1, lr_head=.001 if algorithm == "fedrep_nonlinear" else .01,
                            head_optimizer="adam_reset" if algorithm == "fedrep_nonlinear" else "sgd",
                            lr_schedule="constant", momentum=.9, batch_size=32,
                            rounds=2 if smoke else 200, eval_every=1 if smoke else 10,
                            use_leaf=True))
    return configs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    configs = configurations(args.smoke)
    print(f"Prepared {len(configs)} configurations; {configs[0]['rounds']} rounds each.", flush=True)
    if args.dry_run:
        print(json.dumps(configs, indent=2))
        return
    import torch
    from experiments.run_experiment import run_experiment
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty.strip():
        raise RuntimeError("Tracked source changes detected; commit them before this reproducible run")
    root = Path(args.output_dir)
    manifest = {"revision": revision, "configs": configs, "device": args.device,
                "data_dir": str(Path(args.data_dir).resolve()),
                "torch_version": torch.__version__, "provisional": True}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Output manifest differs; use a fresh output directory")
    elif root.exists() and any(root.iterdir()):
        raise ValueError("Nonempty output directory has no manifest; use a fresh directory")
    else:
        save_json(manifest_path, manifest)
    failures = []
    for index, cfg in enumerate(configs, 1):
        path = (root / cfg['dataset'] / cfg['loss_type'] / cfg['algorithm'] /
                cfg['aggregator'] / cfg['attack'] / f"seed_{cfg['seed']}.json")
        checkpoint = path.with_suffix(".pt")
        if path.exists() and checkpoint.exists():
            saved = json.loads(path.read_text())
            if saved.get("run_config") != cfg:
                raise ValueError(f"Configuration mismatch: {path}")
            print(f"[{index}/{len(configs)}] skip {path}", flush=True)
            continue
        print(f"[{index}/{len(configs)}] run {path}", flush=True)
        started = time.monotonic()
        try:
            result = run_experiment(**cfg, data_dir=args.data_dir, device=args.device,
                                    checkpoint_path=str(checkpoint))
            payload = result.to_dict()
            payload.update(run_config=cfg, source_revision=revision,
                           elapsed_seconds=time.monotonic()-started)
            save_json(path, payload)
        except Exception as error:
            failures.append({"config": cfg, "error": str(error), "traceback": traceback.format_exc()})
            save_json(root / "failures.json", failures)
            traceback.print_exc()
            if args.smoke:
                raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    save_json(root / "failures.json", failures)
    print(f"Grid finished: {len(failures)} failed configurations. Outputs: {root}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
