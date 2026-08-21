#!/bin/bash
#SBATCH --job-name=lambda_gp_diag
#SBATCH --output=outputs/10p/lambda_gp_diagnostic_%a.out
#SBATCH --error=outputs/10p/lambda_gp_diagnostic_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=03-00:00:00
#SBATCH --array=1-4

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

# CLAUDE.md TODO item 20 follow-up (docs/investigations/critic-saturation-diagnostic-2026-08.md
# "Next steps"): tests whether lambda_gp=5 (fixed across every run in this repo so far) is what
# makes degenerate critic collapse a cheap optimum. Mechanistically, a collapsed critic drives
# the Wasserstein term to ~0 and the gradient penalty to ~1, so critic_loss settles at
# ~lambda_gp — a large lambda_gp rewards a flat, input-independent critic for doing nothing.
# This sweep brackets the existing lambda_gp=5 baseline on both sides (0, 0.1, 1, 10) rather
# than re-running 5 itself — that data already exists from
# critic-saturation-diagnostic-kl0.05 (run 19y835y0, 15 epochs, see
# docs/runs/critic-saturation-diagnostic-kl0.05_19y835y0.md), where collapse was fully
# established by epoch 4-5. 8 epochs here gives enough runway to see the same thing (or its
# absence) at roughly half the wall-clock cost of the original diagnostic runs.
# lambda_kl is held fixed at 0.05 (the diagnostic's stable/uneventful arm) to isolate the
# lambda_gp effect from lambda_kl's separately-established effect on post-collapse stability.
# Note: lambda_gp=0 fully disables the gradient penalty (no Lipschitz constraint) — this arm
# may diverge/explode rather than converge or collapse; that's an expected, informative
# possible outcome, not a run failure.
gp_list=(0 0.1 1 10)
lambda_gp=${gp_list[$SLURM_ARRAY_TASK_ID-1]}

# Construct a run name
run_name="lambda-gp-diagnostic-gp${lambda_gp}"

python 10p_train.py \
  --n_critic    4 \
  --lambda_gp   $lambda_gp \
  --lambda_kl   0.05 \
  --lr_gen      5e-6 \
  --lr_critic   5e-5 \
  --temperature 1.0 \
  --n_epochs    8 \
  --batch_size  8 \
  --num_eval_sequences 30 \
  --holdout_size 200 \
  --loss_log_every 25 \
  --seed        89 \
  --run_name    $run_name \
  --wandb_tags  lambda-gp-diagnostic,seeded
