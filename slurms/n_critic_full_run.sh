#!/bin/bash
#SBATCH --job-name=n_c_full
#SBATCH --output=outputs/full/full_n_critic_%a.out
#SBATCH --error=outputs/full/full_n_critic_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=03-00:00:00
#SBATCH --array=1-5

# Keep HF offline flags if you need them:
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# W&B key
# you need to set your apı key from an external slurm script with the line below
# export WANDB_API_KEY=
# then here
source /path/to/env.sh

# Your Conda + PROGRES settings
module load miniforge/24.3.0-0
source activate protgen_env_conda
export PROGRES_DATA_DIR=/gpfs/projects/etur29/ufuk/progres/

# Make Python prints unbuffered so you see them live in your .out file
export PYTHONUNBUFFERED=1

# Map the SLURM array index to your n_critic values
nc_list=(1 2 4 8 12)
n_critic=${nc_list[$SLURM_ARRAY_TASK_ID-1]}

# Construct a run name
run_name="mn5_full_n_critic_${n_critic}"

# Launch training
python fully_masked_train.py \
  --n_critic  $n_critic \
  --lambda_gp 5 \
  --lr_gen    5e-5 \
  --lr_critic 5e-5 \
  --n_epochs  25 \
  --batch_size 8 \
  --run_name  $run_name

