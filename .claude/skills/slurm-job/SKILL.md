---
name: slurm-job
description: Generate a SLURM job script for a ProtGen training or generation run on MN5
---

Generate a SLURM `.sh` script for MN5 based on user-provided parameters.

## Required parameters (ask if missing)
- `run_name` — used in --job-name and wandb run name
- `script` — which Python script: `10p_train.py`, `fully_masked_train.py`, or `generate.py`
- `lr_gen` — generator learning rate (e.g. `5e-5`)
- `lr_crit` — critic learning rate (e.g. `5e-4`)
- `n_critic` — critic steps per generator step (default: 8)
- `n_epochs` — number of epochs (default: 10)
- `batch_size` — default 8 for training, 16 for generation

## SLURM header to use (MN5 standard)
```bash
#!/bin/bash
#SBATCH --job-name=<run_name>
#SBATCH --output=outputs/<run_name>_%j.out
#SBATCH --error=outputs/<run_name>_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --qos=acc_bscsup

module load cuda/11.8
module load conda
conda activate protgen_env_conda

export SOURCE_DIR=/gpfs/projects/etur29/ufuk
```

## Rules
- Always set `--output` and `--error` to `outputs/` dir
- Always export `SOURCE_DIR`
- For generation runs: add `--ckpt_id` and `--num_full`/`--num_seeded` args
- For training: include `--run_name`, `--lr_gen`, `--lr_crit`, `--n_critic`, `--n_epochs`, `--batch_size`
- Write the file to `slurms/<run_name>.sh`
