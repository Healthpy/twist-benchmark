#!/bin/bash
# Multi-world run (pilot or campaign) on one Snellius CPU node; tasks run as parallel processes.
#   sbatch scripts/slurm/multiworld.sbatch experiments/pilot.yaml           (Week-5 pilot)
#   sbatch scripts/slurm/multiworld.sbatch experiments/access_ladder.yaml ladder  (Week-6 ladder)
#   sbatch scripts/slurm/multiworld.sbatch experiments/robustness.yaml robustness (Week-7 robustness)
#   sbatch --cpus-per-task=32 scripts/slurm/multiworld.sbatch experiments/study.yaml study    (Week-8 study)
#   sbatch --cpus-per-task=10 scripts/slurm/multiworld.sbatch experiments/discovery.yaml discovery  (discovery study)
#   sbatch --cpus-per-task=5 scripts/slurm/multiworld.sbatch experiments/bridge.yaml bridge  (Week-9 energy bridge)
# Check the partition, CPU count and time limit against your allocation before submitting.
#SBATCH --job-name=twist-multiworld
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export TWIST_NUM_THREADS=1   # one thread per task process
.venv/bin/twist "${2:-pilot}" --config "${1:-experiments/pilot.yaml}" --jobs "${SLURM_CPUS_PER_TASK}"
