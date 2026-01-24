#!/bin/bash
#SBATCH --job-name=lr_full
#SBATCH --output=outputs/full/full_lrgen_%a_lrcrit_%a.out
#SBATCH --error=outputs/full/full_lrgen_%a_lrcrit_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=03-00:00:00
#SBATCH --array=1-25

# Keep HF offline flags if you need them:
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# W&B key (if you still log runs)
export WANDB_API_KEY=***REMOVED***

# Your Conda + PROGRES settings
module load miniforge/24.3.0-0
source activate protgen_env_conda
export PROGRES_DATA_DIR=/gpfs/projects/etur29/ufuk/progres/
export PYTHONUNBUFFERED=1

# Convert SLURM_ARRAY_TASK_ID (1�~@~S25) into zero-based index (0�~@~S24):
task_index=$(( SLURM_ARRAY_TASK_ID - 1 ))

# Compute indices into each list:
#   gen_idx = floor(task_index / 5)
#   crit_idx = task_index % 5
gen_idx=$(( task_index / 5 ))
crit_idx=$(( task_index % 5 ))

lr_gen=${lr_gen_list[$gen_idx]}
lr_critic=${lr_crit_list[$crit_idx]}

# Make Python prints unbuffered so you see them live in your .out file
# Define both LR lists (5 each):
lr_gen_list=(1e-6 5e-6 1e-5 5e-5 5e-4)
lr_crit_list=(1e-6 5e-6 1e-5 5e-5 5e-4)

# Convert SLURM_ARRAY_TASK_ID (1�~@~S25) into zero-based index (0�~@~S24):
task_index=$(( SLURM_ARRAY_TASK_ID - 1 ))

# Compute indices into each list:
#   gen_idx = floor(task_index / 5)
#   crit_idx = task_index % 5
gen_idx=$(( task_index / 5 ))
crit_idx=$(( task_index % 5 ))

lr_gen=${lr_gen_list[$gen_idx]}
lr_critic=${lr_crit_list[$crit_idx]}

# Construct a run name
run_name="full_nc8_lrgen_${lr_gen}_lrcrit_${lr_critic}"

# Launch training
python fully_masked_train.py \
  --n_critic  8 \
  --lambda_gp 5 \
  --lr_gen    $lr_gen \
  --lr_critic $lr_critic \
  --n_epochs  10 \
  --batch_size 16 \
  --run_name  $run_name
