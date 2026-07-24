#!/bin/bash
#SBATCH --job-name=kl_sweep_p1
#SBATCH --output=outputs/10p/kl_sweep_p1_%a.out
#SBATCH --error=outputs/10p/kl_sweep_p1_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=03-00:00:00
#SBATCH --array=1-5

# Keep HF offline flags if you need them:
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Load secrets (WANDB_API_KEY, etc.) from the repo's .env file — gitignored, never committed.
if [ -f "$SLURM_SUBMIT_DIR/.env" ]; then
    set -a
    source "$SLURM_SUBMIT_DIR/.env"
    set +a
fi

# Your Conda + PROGRES settings
export SOURCE_DIR=/gpfs/projects/etur29/ufuk
source /gpfs/projects/etur29/ufuk/envs/protgen-gan/bin/activate
export PROGRES_DATA_DIR=/gpfs/projects/etur29/ufuk/progres/

# Make Python prints unbuffered so you see them live in your .out file
export PYTHONUNBUFFERED=1

# Phase 1 screening grid — see docs/superpowers/specs/2026-07-24-lambda-kl-sweep-design.md
# Map the SLURM array index to a lambda_kl value.
kl_list=(0 0.005 0.01 0.05 0.1)
lambda_kl=${kl_list[$SLURM_ARRAY_TASK_ID-1]}

# Construct a run name
run_name="kl-sweep-p1-kl${lambda_kl}"

# Launch training — fixed params per the sweep design: n_critic=4, lr_gen=5e-6,
# lr_critic=5e-5, temperature=1.0 (pinned), full dataset, 5 epochs.
python 10p_train.py \
  --n_critic    4 \
  --lambda_gp   5 \
  --lambda_kl   $lambda_kl \
  --lr_gen      5e-6 \
  --lr_critic   5e-5 \
  --temperature 1.0 \
  --n_epochs    5 \
  --batch_size  8 \
  --run_name    $run_name
