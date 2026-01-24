#!/bin/bash
#SBATCH --job-name=generate_%a
#SBATCH --output=outputs/generate/%a.out
#SBATCH --error=outputs/generate/%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=03-00:00:00
#SBATCH --array=1-6
#SBATCH --chdir=/home/hu/hu733216/protgen/gan/

set -euo pipefail


# --- env ---
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export TORCH_USE_CUDA_DSA=0
export CUDA_LAUNCH_BLOCKING=0

module load miniforge/24.3.0-0
source activate protgen_env_conda

# --- parameters ---
# ckpt_id must match generate.py:
# - trained: "run_name/epoch_X"
# - finetuned baseline: "finetuned_protbert"
# - base baseline: "protbert_base"
ckpt_id_list=(
  "10p_freeze_gen_5e-6_crit_5e-5/epoch_13"
  "10p_nc8_lrgen_5e-6_lrcrit_5e-5/epoch_5"
  "full_freeze_gen_5e-5_crit_5e-4/epoch_14"
  "full_nc8_lrgen_5e-5_lrcrit_5e-4/epoch_7"
  "finetuned_protbert"
  "protbert_base"
)

# generation split per case: 10k full, 10k seeded
num_full_list=( 10000 10000 10000 10000 10000 10000 )
num_seeded_list=( 10000 10000 10000 10000 10000 10000 )

# data sources
DATASET_PATH="./data/dnmt_unformatted.txt"
SEED_FILE="./data/dnmt_unformatted.txt"   # if you don't have a held-out file yet, set this equal to DATASET_PATH

idx=$(( SLURM_ARRAY_TASK_ID - 1 ))

# safety check
if (( idx < 0 || idx >= ${#ckpt_id_list[@]} )); then
  echo "Index $idx out of range"; exit 1
fi

ckpt_id="${ckpt_id_list[$idx]}"
num_full="${num_full_list[$idx]}"
num_seeded="${num_seeded_list[$idx]}"

echo "[INFO] idx=$idx ckpt_id=$ckpt_id full=$num_full seeded=$num_seeded"

# --- launch ---
python ./generate.py \
  --ckpt_id "$ckpt_id" \
  --dataset_path "$DATASET_PATH" \
  --seed_file "$SEED_FILE" \
  --num_full "$num_full" \
  --num_seeded "$num_seeded" \
  --out_dir "./results/generation_23_01_25/" \
  --batch_size 64 \
  --device cuda

