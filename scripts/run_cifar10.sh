#!/usr/bin/env bash
set -euo pipefail
python scripts/run_paper_experiments.py --datasets cifar10 "$@"
