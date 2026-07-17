#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --output=outputs/smoke_test_%j.out
#SBATCH --error=outputs/smoke_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:60:00

# Keep HF offline flags if you need them:
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Load secrets (WANDB_API_KEY, etc.) from the repo's .env file — gitignored, never committed.
if [ -f "$SLURM_SUBMIT_DIR/.env" ]; then
    set -a
    source "$SLURM_SUBMIT_DIR/.env"
    set +a
fi

# protgen-gan env from conda-pack (no module load needed — self-contained)
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

run_name="smoke_test-2epoch"

# Toy run: 2 epoch, tiny batch/dataset, ESMFold load, KL anchor.
# Just proves the packed env can run the real training entrypoint end-to-end on a GPU node.
python 10p_train.py \
  --run_name $run_name \
  --n_epochs 2 \
  --batch_size 4 \
  --n_critic 4 \
  --num_eval_sequences 4 \
  --max_train_seqs 200 \
  --lambda_kl 0.01
