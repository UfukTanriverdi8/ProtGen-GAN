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
# FILL IN before submitting: replace with the top-2 lambda_kl values selected
# from Phase 1 (docs/sweeps/lambda-kl-sweep-2026-07.md's "Winner selection
# reasoning" section) once those runs finish. Placeholder values below are
# NOT real winners — do not submit as-is.
kl_list=(FILL_ME_1 FILL_ME_2)
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
  --run_name    $run_name
