#!/bin/bash
#SBATCH --job-name=long_10p
#SBATCH --output=outputs/10p/long_run/8_n_critic_10p.out
#SBATCH --error=outputs/10p/long_run/8_n_critic_10p.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=03-00:00:00

# Keep HF offline flags if you need them:
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# W&B key (if you still log runs)
export WANDB_API_KEY=***REMOVED***

# Your Conda + PROGRES settings
module load miniforge/24.3.0-0
source activate protgen_env_conda
export PROGRES_DATA_DIR=/gpfs/projects/etur29/ufuk/progres/

# Make Python prints unbuffered so you see them live in your .out file
export PYTHONUNBUFFERED=1

# Map the SLURM array index to your n_critic values
# nc_list=(1 2 4 8 12)
# n_critic=${nc_list[$SLURM_ARRAY_TASK_ID-1]}

# Construct a run name
run_name="mn5_10p_n_critic_8_long"

# Launch training
python 10p_train.py \
  --n_critic  8 \
  --lambda_gp 5 \
  --lr_gen    5e-5 \
  --lr_critic 5e-5 \
  --n_epochs  100 \
  --batch_size 16 \
  --run_name  $run_name
