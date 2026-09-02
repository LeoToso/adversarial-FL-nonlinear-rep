#!/usr/bin/env python3
"""Robust H=10/20/50 scaling grid for Collins CIFAR-10 and School."""
import argparse, itertools, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from experiments.run_experiment import run_experiment

PROFILES={
 "cifar10": dict(dataset="cifar10_collins", alpha=1.0, n_clients=105,
                 n_byzantine=5, loss_type="cross_entropy", repr_dim=64,
                 head_steps=10, lr=.01, lr_head=.01, momentum=.5,
                 rounds=100, batch_size=10),
 "school": dict(dataset="school", alpha=1.0, n_clients=144,
                n_byzantine=5, loss_type="least_squares", repr_dim=64,
                head_steps=20, lr=.05, lr_head=.01, momentum=.9,
                rounds=500, batch_size=32),
}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--dataset",choices=PROFILES,required=True)
 p.add_argument("--honest_clients",nargs="+",type=int,default=[10,20,50])
 p.add_argument("--seeds",nargs="+",type=int,default=[42,123,456,789,1024])
 p.add_argument("--aggregators",nargs="+",default=["NNM+TrMean","NNM+Krum"])
 p.add_argument("--attacks",nargs="+",default=["ALIE","Mimic"])
 p.add_argument("--device",default="cpu"); p.add_argument("--data_dir",default="./data")
 p.add_argument("--output_dir",default="results/client_scaling")
 p.add_argument("--eval_every",type=int,default=10); p.add_argument("--rounds",type=int)
 p.add_argument("--num_shards",type=int,default=1); p.add_argument("--shard_index",type=int,default=0)
 p.add_argument("--overwrite",action="store_true"); a=p.parse_args()
 grid=list(itertools.product(a.honest_clients,["baseline","fedrep_nonlinear"],a.aggregators,a.attacks,a.seeds))
 jobs=[x for i,x in enumerate(grid) if i%a.num_shards==a.shard_index]
 print(f"Prepared {len(jobs)} of {len(grid)} configurations",flush=True)
 for h,alg,agg,attack,seed in jobs:
  path=Path(a.output_dir)/a.dataset/f"honest_{h}"/alg/agg/attack/f"seed_{seed}.json"; ck=path.with_suffix(".pt")
  if path.exists() and ck.exists() and not a.overwrite: print("skip",path); continue
  cfg=dict(PROFILES[a.dataset]); cfg.update(algorithm=alg,aggregator=agg,attack=attack,
      seed=seed,active_honest_clients=h,device=a.device,data_dir=a.data_dir,
      eval_every=a.eval_every,checkpoint_path=str(ck))
  if a.rounds: cfg["rounds"]=a.rounds
  print("run",path,flush=True); result=run_experiment(**cfg); obj=result.to_dict()
  obj["dataset"]=a.dataset; obj["honest_per_round"]=h; obj["byzantine_per_round"]=5
  path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(".tmp")
  with tmp.open("w") as f: json.dump(obj,f,indent=2,allow_nan=False)
  os.replace(tmp,path)
if __name__=="__main__": main()
