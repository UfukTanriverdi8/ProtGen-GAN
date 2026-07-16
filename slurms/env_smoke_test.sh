#!/bin/bash
#SBATCH --job-name=env_smoke_test
#SBATCH --output=outputs/env_smoke_test_%j.out
#SBATCH --error=outputs/env_smoke_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

# Keep HF offline flags if you need them:
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# W&B key
# you need to set your apı key from an external slurm script with the line below
# export WANDB_API_KEY=
# then here
source /path/to/env.sh

# ProtGen conda-pack env (no module load needed — self-contained)
export SOURCE_DIR=/gpfs/projects/etur29/ufuk
source /gpfs/projects/etur29/ufuk/envs/protgen-gan/bin/activate
export PROGRES_DATA_DIR=/gpfs/projects/etur29/ufuk/progres/

# Make Python prints unbuffered so you see them live in your .out file
export PYTHONUNBUFFERED=1

echo "=== env sanity check ==="
which python
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda avail:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "========================"

run_name="env_smoke_test"

# Toy run: 1 epoch, tiny batch/dataset, no ESMFold load, no KL anchor.
# Just proves the packed env can run the real training entrypoint end-to-end on a GPU node.
python 10p_train.py \
  --run_name $run_name \
  --n_epochs 1 \
  --batch_size 4 \
  --n_critic 4 \
  --num_eval_sequences 0 \
  --max_train_seqs 200 \
  --lambda_kl 0.0
