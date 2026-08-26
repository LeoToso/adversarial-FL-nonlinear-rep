# Paper experiment protocol

This repository implements the two empirical settings in the paper draft:

1. `baseline`: adversarial federated learning with one shared predictor.
2. `fedrep_nonlinear`: a shared nonlinear representation and private linear
   client heads. Heads are fitted locally before the independent backbone
   gradient batch, and only backbone gradients are robustly aggregated.

## Experiment grid

- Datasets: CIFAR-10, natural-writer LEAF FEMNIST, and the ILEA School Exam
  Score regression dataset.
- Classification objectives: cross entropy and the paper's multiclass squared
  loss, `mean(||one_hot(y) - scores||_2^2)`, without softmax.
- Regression objective: mean scalar squared error.
- Aggregators: NNM+TrMean, NNM+GM, and NNM+Krum.
- Attacks: Sign Flipping and Inner Product Manipulation.
- Algorithms: baseline and nonlinear FedRep.
- Default seeds: 42, 123, 456, 789, and 1024.

## Data preparation

CIFAR-10 is downloaded by torchvision.

For FEMNIST, generate or download the LEAF FEMNIST train/test JSON files and
place them under:

```text
data/femnist/train/*.json
data/femnist/test/*.json
```

Train and test records are matched by writer ID. A deterministic subset of
writers is selected for each seed, and the output dimension is inferred from
the LEAF labels (normally 62 FEMNIST classes).

The School loader downloads the canonical `school.mat` file from the MALSAR
repository into `data/school/` on first use. Each school is one natural client.
All 139 school tasks are used by default, together with 35 simulated Byzantine
workers.

## Running

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run one dataset's complete five-seed grid:

```bash
bash scripts/run_cifar10.sh --device cuda
bash scripts/run_femnist_leaf.sh --device cuda
bash scripts/run_school.sh --device cuda
```

For a quick one-seed smoke run:

```bash
python scripts/run_paper_experiments.py \
  --datasets cifar10 --seeds 42 --rounds 2 --eval_every 1
```

Preview the grid without loading PyTorch or any datasets:

```bash
python scripts/run_paper_experiments.py --seeds 42 --dry_run
```

Completed configurations are skipped unless `--overwrite` is supplied.

## Results

Every configuration/seed is saved independently under:

```text
results/paper/<dataset>/<loss>/<algorithm>/<aggregator>/<attack>/seed_<seed>.json
```

Each file contains:

- `train_history`: training objective at every communication round;
- `history`: held-out objective and accuracy/MSE at evaluation rounds;
- the complete dataset, loss, algorithm, attack, aggregation, and seed metadata.

Generate mean and one-standard-deviation loss curves across seeds with:

```bash
python scripts/plot_paper_losses.py
```

Plots are saved as PDF and PNG under `results/paper_figures/`.
