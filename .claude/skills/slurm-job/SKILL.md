---
name: slurm-job
description: Generate a SLURM job script for a ProtGen-GAN training or generation run on MN5
---

Generate a SLURM `.sh` script for MN5 based on user-provided parameters. Base every script on
one of the two reference templates below — don't improvise the header or env block.

## Ask first (if not given)

- **Script:** `10p_train.py` (seeded), `fully_masked_train.py` (blind), or `generate.py`
- **Single run or array job (sweep)?** If the user wants to test more than one value of any
  parameter (lr, n_critic, lambda_kl, ckpt_id, ...), it's an array job — this is the common
  case for anything but a single confirmed-config run. Don't default to single-run without
  asking; every current sweep/grid-search script in this repo is an array job.
- Training-specific: `run_name`, `lr_gen`, `lr_critic`, `n_critic` (default 4), `n_epochs`,
  `batch_size` (default 8), `lambda_kl` (default 0.01), `temperature` (default 1.0),
  `wandb_tags` (comma-separated, optional but recommended for anything that's part of a sweep
  — e.g. `kl-sweep-p1`)
- Generation-specific: `ckpt_id` (`<run_name>/epoch_<N>`, `finetuned_protbert`, or
  `protbert_base`), `num_full`, `num_seeded`

## Reference templates

Match these exactly — don't invent a different env block or output path convention.

### Training (10p_train.py / fully_masked_train.py), array job

```bash
#!/bin/bash
#SBATCH --job-name=<short_name>
#SBATCH --output=outputs/<10p|full>/<short_name>_%a.out
#SBATCH --error=outputs/<10p|full>/<short_name>_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=03-00:00:00
#SBATCH --array=1-<N>

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

# Map the SLURM array index to your swept parameter
<param>_list=(<value1> <value2> ...)
<param>=${<param>_list[$SLURM_ARRAY_TASK_ID-1]}

# Construct a run name
run_name="<prefix>-<param>${<param>}"

python <10p_train.py|fully_masked_train.py> \
  --n_critic    <n_critic> \
  --lambda_gp   5 \
  --lambda_kl   <lambda_kl> \
  --lr_gen      <lr_gen> \
  --lr_critic   <lr_critic> \
  --temperature <temperature> \
  --n_epochs    <n_epochs> \
  --batch_size  <batch_size> \
  --num_eval_sequences 30 \
  --run_name    $run_name \
  --wandb_tags  <tag>
```

For a **single run** (no sweep), drop `#SBATCH --array=`, drop the `_list`/index-mapping
block, and hardcode every value directly in the `python` call — see `slurms/long_10p_run.sh`
for the exact pattern.

### Generation (generate.py)

```bash
#!/bin/bash
#SBATCH --job-name=generate_%a
#SBATCH --output=outputs/generate/%a.out
#SBATCH --error=outputs/generate/%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=03-00:00:00
#SBATCH --array=1-<N>
#SBATCH --chdir=/home/hu/hu733216/protgen/gan/

set -euo pipefail

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true

export SOURCE_DIR=/gpfs/projects/etur29/ufuk
source /gpfs/projects/etur29/ufuk/envs/protgen-gan/bin/activate

ckpt_id_list=(<...>)
num_full_list=(<...>)
num_seeded_list=(<...>)
idx=$(( SLURM_ARRAY_TASK_ID - 1 ))

python ./generate.py \
  --ckpt_id "${ckpt_id_list[$idx]}" \
  --dataset_path "./data/dnmt_unformatted.txt" \
  --seed_file "./data/dnmt_unformatted.txt" \
  --num_full "${num_full_list[$idx]}" \
  --num_seeded "${num_seeded_list[$idx]}" \
  --out_dir "./eval_sequences/120k_generation_output/" \
  --batch_size 64 \
  --device cuda
```

Generation runs don't source `.env` or export `PROGRES_DATA_DIR` — no wandb logging
(`WANDB_DISABLED=true`) and no progres eval in the generation path itself.

## Hard rules (get these wrong and the job either fails to submit or corrupts results)

- **Never use `module load` or `conda activate`.** The env is a `conda-pack`-packed
  environment activated via `source .../envs/protgen-gan/bin/activate` — mixing that with
  `conda activate`/`deactivate` on MN5 partially clobbers `CONDA_PREFIX`/`PATH` and
  deactivation silently fails. This is the single most common way an old script goes stale.
- **`--account`/`--qos` are mandatory but go on the `sbatch` command line, not in the script**
  — MN5 hard-errors without them ("No account specified"/"No QoS specified"). This repo's
  convention: `sbatch --account=etur29 --qos=acc_ehpc slurms/<path>/<script>.sh`. After
  writing the script, always tell the user the exact submit command — don't assume they'll
  remember the flags.
- **`acc_ehpc`'s wallclock cap is 72h** — matches this repo's standard `--time=03-00:00:00`
  exactly, i.e. already at the ceiling. If a run needs longer, it needs a different qos, not
  a bigger `--time` value (which MN5 will simply reject). For quick smoke tests, `acc_debug`
  (2h limit, 8 nodes max) is the right qos instead of shortening `--time` under `acc_ehpc`.
  Confirm timing math against the actual queue limits before writing a `--time` value blindly
  for anything unusual (very long `n_epochs`, full dataset + many eval sequences, etc.).
- **1 GPU = 20 CPU** on the ACC partition (`--gres=gpu:1` pairs with `--cpus-per-task=20`;
  each ACC node has 80 CPU / 4 GPU) — this repo's standard is always 1 GPU per task, don't
  request more without the user asking for it and don't change the CPU ratio.
- **Submission is assumed to happen from the repo root** (`$SLURM_SUBMIT_DIR` is what `.env`
  sourcing relies on) for training scripts. Generation scripts instead hardcode an absolute
  `--chdir=`. Match whichever convention the reference template for that script type uses —
  don't mix them.
- **Output path is nested by script type**, not flat: `outputs/10p/` for `10p_train.py`,
  `outputs/full/` for `fully_masked_train.py`, `outputs/generate/` for `generate.py`.
- **File location:** array-job/sweep scripts go in `slurms/sweeps/<name>.sh`; standalone
  single-run scripts go in `slurms/<name>.sh`.
- If this run is part of a sweep, pass `--wandb_tags` (e.g. `kl-sweep-p1`) so results are
  filterable in the wandb dashboard without parsing run names — see
  `docs/sweeps/lambda-kl-sweep-2026-07.md` for the convention this supports.

## After writing the file

Tell the user the exact submit command, e.g.:

```
sbatch --account=etur29 --qos=acc_ehpc slurms/sweeps/<name>.sh
```

If this script documents results anywhere (a sweep), point to
`docs/RUN_DOCUMENTATION.md`'s convention and mention the `wandb-run-reviewer` subagent is
available to document each run once synced.