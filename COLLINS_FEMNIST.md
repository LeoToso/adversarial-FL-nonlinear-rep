# Collins et al. FEMNIST benchmark

This benchmark is deliberately separate from `femnist`, which remains the
canonical 62-class natural-writer LEAF task.  `femnist_collins` follows the
ICML 2021 comparison protocol: 10 lowercase letters, 150 synthetic clients,
3 classes per client, 15 honest clients sampled per round, batch size 10,
SGD with learning rate 0.01 and momentum 0.5, 10 head epochs, 5 representation
epochs, 200 rounds, and mean local accuracy over the final 10 rounds.

The clean reference target is 78.56% for FedRep and 51.64% for FedAvg.  The
paper reports the mean of five runs but no standard deviation.

## Prepare data

Use the complete LEAF corpus already located in `data/femnist`:

```bash
python scripts/prepare_collins_femnist.py \
  --leaf_dir data/femnist \
  --output_dir data/femnist_collins \
  --seed 42 \
  --published_compatibility
```

The compatibility partition matches the paper's sample statistics. Because
the available lowercase class pools are finite, exhausted pools are cycled;
the generated metadata records the exact number and fraction of reused
assignments. Omit `--published_compatibility` for a strictly disjoint ablation,
which may have fewer than 148 mean training samples per client.

## Smoke test

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_collins_femnist.py \
  --mode clean \
  --algorithms fedrep \
  --seeds 42 \
  --rounds 2 \
  --eval_every 1 \
  --device cuda \
  --output_dir results/femnist_collins_smoke
```

## Clean five-run calibration

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_collins_femnist.py \
  --mode clean \
  --algorithms fedavg fedrep \
  --seeds 42 123 456 789 1024 \
  --device cuda \
  --output_dir results/femnist_collins_clean \
  2>&1 | tee logs/femnist_collins_clean.log
```

## Byzantine extension

This retains 15 honest participants per round and appends 5 Byzantine
messages, hence 20 received updates and a 25% Byzantine message fraction.

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_collins_femnist.py \
  --mode robust \
  --algorithms fedavg fedrep \
  --aggregators 'NNM+TrMean' 'NNM+Krum' \
  --attacks SignFlipping InnerProductManipulation \
  --seeds 42 123 456 789 1024 \
  --byzantine_per_round 5 \
  --device cuda \
  --output_dir results/femnist_collins_robust \
  2>&1 | tee logs/femnist_collins_robust.log
```

The released FedRep implementation applied softmax before
`CrossEntropyLoss`; this behavior is enabled by default for reproduction.
Use `--standard_logits` only as a separately labelled ablation.

## ALIE attack

ALIE (A Little Is Enough) is distinct from Inner Product Manipulation. For
honest client vectors with coordinate-wise mean `mu` and population standard
deviation `sigma`, it submits `mu + tau * sigma`. The default is
`tau=1.5`, matching ByzFL. Run only ALIE with:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_collins_femnist.py \
  --mode robust \
  --algorithms fedavg fedrep \
  --aggregators 'NNM+TrMean' 'NNM+Krum' \
  --attacks ALIE \
  --attack_tau 1.5 \
  --seeds 42 123 456 789 1024 \
  --byzantine_per_round 5 \
  --device cuda \
  --output_dir results/femnist_collins_alie_5seeds
```
