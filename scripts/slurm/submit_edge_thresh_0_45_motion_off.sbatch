#!/bin/bash
#SBATCH -p gpu_batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -t 0-05:00:00
#SBATCH -J edge045moff
#SBATCH -o /home/mc46451/biohub-cell-tracking/logs/slurm/edge_thresh_0_45_motion_off_%j.out
#SBATCH -e /home/mc46451/biohub-cell-tracking/logs/slurm/edge_thresh_0_45_motion_off_%j.err

set -euo pipefail
# Launcher only: the worker script stages to /tmp and runs its own srun GPU allocation.
bash /home/mc46451/biohub-cell-tracking/scripts/slurm/run_edge_thresh_0_45_motion_off.sh
