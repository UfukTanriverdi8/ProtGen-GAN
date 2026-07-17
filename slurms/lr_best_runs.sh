#!/bin/bash
#SBATCH --job-name=lr_sweep_%a
#SBATCH --output=outputs/lr_best_runs/%a.out
#SBATCH --error=outputs/lr_best_runs/%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=03-00:00:00
#SBATCH --array=1-7           # 7 distinct runs

# ---------- environment ----------
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Load secrets (WANDB_API_KEY, etc.) from the repo's .env file — gitignored, never committed.
if [ -f "$SLURM_SUBMIT_DIR/.env" ]; then
    set -a
    source "$SLURM_SUBMIT_DIR/.env"
    set +a
fi

export SOURCE_DIR=/gpfs/projects/etur29/ufuk
source /gpfs/projects/etur29/ufuk/envs/protgen-gan/bin/activate
export PROGRES_DATA_DIR=/gpfs/projects/etur29/ufuk/progres/
export PYTHONUNBUFFERED=1

# ---------- lookup tables ----------
mode_list=( full full 10p 10p 10p 10p 10p )
lr_gen_list=( 1e-5 5e-5 5e-5 5e-6 5e-6 5e-5 5e-5 )
lr_crit_list=( 5e-5 5e-4 1e-6 1e-5 5e-5 5e-6 5e-4 )

idx=$(( SLURM_ARRAY_TASK_ID - 1 ))

mode=${mode_list[$idx]}
lr_gen=${lr_gen_list[$idx]}
lr_crit=${lr_crit_list[$idx]}

# choose script & run-name
if [[ "$mode" == "full" ]]; then
  train_script="fully_masked_train.py"
else
  train_script="10p_train.py"
fi

run_name="${mode}_freeze_gen_${lr_gen}_crit_${lr_crit}"

# ---------- launch ----------
python $train_script \
  --n_critic   8 \
  --lambda_gp  5 \
  --lr_gen     $lr_gen \
  --lr_critic  $lr_crit \
  --n_epochs   20 \
  --batch_size 16 \
  --run_name   $run_name

