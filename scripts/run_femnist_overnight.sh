#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1
mkdir -p logs
python scripts/run_femnist_overnight.py --device cuda --smoke \
  --output_dir results/femnist_adam_reset_v6_smoke \
  2>&1 | tee -a logs/femnist_adam_reset_v6_smoke.log
python scripts/run_femnist_overnight.py --device cuda \
  --output_dir results/femnist_adam_reset_v6 \
  2>&1 | tee -a logs/femnist_adam_reset_v6.log
python scripts/summarize_paper_results.py \
  --results_dir results/femnist_adam_reset_v6/femnist \
  --output_dir results/femnist_adam_reset_v6_tables \
  --expected_seeds 42 123 456 789 1024
