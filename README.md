# Experiments

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
- Default grid aggregators: NNM+TrMean and NNM+Krum.
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
writers is selected for each seed. LEAF uses a fixed 62-class output space,
even when the selected training writers omit some labels. The separate legacy
EMNIST-Letters proxy uses 26 outputs. LEAF pixels must already be scaled to
[0, 1]; the loader validates shapes, finite pixels, and labels before normalizing.

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

## Correctness checks and FEMNIST diagnostics

Run the offline regression tests (no datasets or GPU required):

```bash
python -m unittest discover -s tests -v
```

The alternating FedRep phases clear stale gradients and clip only the active
head or backbone parameters. CIFAR-10/100 personalized held-out subsets use
deterministic transforms, while training retains random crops and flips.
These held-out subsets are still drawn from the training partition; they are
not the official CIFAR test set. The optional global probe remains separate.

Before another full FEMNIST grid, run a data check, a fixed-training-batch fit
(a diagnostic using Adam, not the federated optimizer), and clean f=0
baseline/FedRep experiments for both objectives:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/diagnose_femnist.py \
  --device cuda --seed 42 --rounds 100 \
  --output_dir results/femnist_diagnostics_v2
```

The clean runs use the same ten writers, representation size, and federated
learning rates as the paper grid, but no Byzantine vectors and simple averaging.
The script stops if the fixed-batch fit fails to reach 95% training accuracy;
this is a debugging gate, not a promised held-out accuracy. Use `--mode data`,
`--mode overfit`, or `--mode clean` to run stages separately. Each invocation
requires a fresh output directory and preserves previous results. JSON logs and
final clean-run checkpoints are saved there. This does not add validation-based
checkpoint selection or resumable training checkpoints.

Keep all pre-fix results in their original directories. Use new directories for
corrected runs: the gradient update and CIFAR evaluation protocol have changed,
so their old and new results must not be pooled into a single five-seed summary.

For the next training-only optimization diagnostic:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/diagnose_femnist_optimization.py \
  --device cuda --seed 42 --rounds 100 --probe_steps 200 \
  --output_dir results/femnist_optimization_v3
```

This compares 10 versus 50 head steps for both losses, leaving other trainer
settings unchanged. It records raw backbone gradient norms, server update norms,
and per-writer training prediction histograms. At the end, copies of each head
are fitted with SGD and Adam on cached, dropout-free training features. These
full-writer probes differ from minibatch federated training and do not guarantee
linear separability or generalization. Test metrics are not used for selection.
Each case saves incremental JSON diagnostics and a final model (not a resumable
checkpoint). The script refuses an existing output seed directory. GPU gradient
instrumentation adds synchronization overhead; do not use its timing as a speed
benchmark. Production trainers and previous results are not modified.
