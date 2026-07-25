#!/bin/bash
#SBATCH --job-name=kl_sweep_p2
#SBATCH --output=outputs/10p/kl_sweep_p2_%a.out
#SBATCH --error=outputs/10p/kl_sweep_p2_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=03-00:00:00
#SBATCH --array=1-2

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

# Phase 2 confirmation — see docs/sweeps/lambda-kl-sweep-2026-07.md
# Top-2 from Phase 1 (see that doc's "Winner selection reasoning"): 0.005 is
# the clear front-runner (stable-to-improving on all 4 quality metrics); 0.05
# narrowly beat 0.1 on composite score but by a thin margin worth re-checking
# at this longer horizon.
kl_list=(0.005 0.05)
lambda_kl=${kl_list[$SLURM_ARRAY_TASK_ID-1]}

# Construct a run name
run_name="kl-sweep-p2-kl${lambda_kl}"

# Launch training — same fixed params as Phase 1, longer horizon to confirm stability.
python 10p_train.py \
  --n_critic    4 \
  --lambda_gp   5 \
  --lambda_kl   $lambda_kl \
  --lr_gen      5e-6 \
  --lr_critic   5e-5 \
  --temperature 1.0 \
  --n_epochs    15 \
  --batch_size  8 \
  --num_eval_sequences 30 \
  --run_name    $run_name \
  --wandb_tags  kl-sweep-p2
