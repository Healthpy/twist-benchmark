#!/bin/bash
# Build the full TWIST suite on a Snellius CPU node:  sbatch scripts/slurm/build_suite.sbatch
#SBATCH --job-name=twist-suite
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export TWIST_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
# export TWIST_SUITE_SECRET=...   # set outside the repository to make hidden splits secret
.venv/bin/twist build-suite --config suites/v1.yaml
.venv/bin/twist check-worlds artifacts/suites/twist_v1
