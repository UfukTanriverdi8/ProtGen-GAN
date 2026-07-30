#!/bin/bash
#SBATCH --job-name=kl_sweep_p2_retry
#SBATCH --output=outputs/10p/kl_sweep_p2_retry_%a.out
#SBATCH --error=outputs/10p/kl_sweep_p2_retry_%a.err
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

# Phase 2 RETRY — see docs/sweeps/lambda-kl-sweep-2026-07.md's 2026-07-30 correction
# note. The original Phase 2 run's quality metrics (plddt/scAccuracy/progres/
# pairwise_tm) were measured under an uncontrolled random temperature
# (generate_fake_sequences ignored --temperature, fixed 2026-07-30) — this
# re-run repeats both arms with that bug fixed. Also gives kl0.05 (the current
# pick) an actual saved checkpoint, since neither original Phase 2 run's
# checkpoint made it to disk (MN5 gpfs_projects quota exceeded).
kl_list=(0.005 0.05)
lambda_kl=${kl_list[$SLURM_ARRAY_TASK_ID-1]}

# Construct a run name
run_name="kl-sweep-p2-retry-kl${lambda_kl}"

# Launch training — same fixed params as the original Phase 2 run.
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
  --wandb_tags  kl-sweep-p2,kl-sweep-p2-retry
