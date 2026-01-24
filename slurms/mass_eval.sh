#!/bin/bash
#SBATCH --job-name=masseval
#SBATCH --output=outputs/masseval_%A_%a.out
#SBATCH --error=outputs/masseval_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=03-00:00:00
#SBATCH --chdir=/home/hu/hu733216/protgen/gan/
#SBATCH --array=0-5

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
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export PROGRES_DATA_DIR=/gpfs/projects/etur29/ufuk/progres/

# --- inputs ---
CSV_IN="eval_results/120k_csv/120k_eval_seqs_prefiltered.csv"

# Each array task handles a slice of rows.
# Tune ROWS_PER_TASK to your dataset size and desired parallelism.
ROWS_PER_TASK=15000

START=$(( SLURM_ARRAY_TASK_ID * ROWS_PER_TASK ))
END=$(( START + ROWS_PER_TASK ))

# Unique output per task (avoid write collisions)
OUT_CSV="eval_results/masseval_120k_part_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

# Unique run name base (your code will also append _aXXXX automatically if you added that)
RUN_NAME="masseval_120k_${SLURM_ARRAY_JOB_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}: rows [${START}, ${END})"
echo "Input:  ${CSV_IN}"
echo "Output: ${OUT_CSV}"
echo "Run:    ${RUN_NAME}"

python ./run_mass_eval.py \
  --csv "${CSV_IN}" \
  --out "${OUT_CSV}" \
  --batch_size 16 \
  --run_name "${RUN_NAME}" \
  --start "${START}" \
  --end "${END}"
