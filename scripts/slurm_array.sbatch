#!/bin/bash
# Slurm job array: one run per GPU. Submit from the repository root:  sbatch scripts/slurm_array.sbatch
# Re-submitting is safe: finished runs are skipped. Run scripts/setup.sh first.
#SBATCH --job-name=dgtf_r1
#SBATCH --array=0-121
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dgtf_%A_%a.out
##SBATCH --partition=gpu           # set your GPU partition if required
mkdir -p logs
# module load cuda; conda activate dgtf     # adapt to your cluster
python scripts/run_experiments.py --task "$SLURM_ARRAY_TASK_ID"
