#!/bin/bash
#SBATCH --job-name=kl_sat_diag
#SBATCH --output=outputs/10p/kl_saturation_diagnostic_%a.out
#SBATCH --error=outputs/10p/kl_saturation_diagnostic_%a.err
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

# CLAUDE.md TODO item 20 diagnostic: re-runs the lambda_kl sweep's two Phase 2 arms
# (docs/sweeps/lambda-kl-sweep-2026-07.md) with the new held-out critic validation
# tooling (--holdout_size, --loss_log_every) enabled. The Phase 2 "0.05 wins" conclusion
# was decided using gen_grad_norm/quality metrics measured while the critic was already
# saturating in most arms — if that saturation is degenerate collapse rather than genuine
# convergence, the critic's signal during that period is suspect and the winner pick
# itself may need re-examination. Running both arms side by side (not just 0.05 alone)
# tests whether collapse is universal (architectural issue, unrelated to lambda_kl) or
# lambda_kl-dependent (original sweep conclusion holds on firmer ground). Same fixed
# params as Phase 2 / the Phase 2 retry.
kl_list=(0.005 0.05)
lambda_kl=${kl_list[$SLURM_ARRAY_TASK_ID-1]}

# Construct a run name
run_name="kl-saturation-diagnostic-kl${lambda_kl}"

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
  --holdout_size 200 \
  --loss_log_every 25 \
  --seed        89 \
  --run_name    $run_name \
  --wandb_tags  item-20,saturation-diagnostic,seeded
