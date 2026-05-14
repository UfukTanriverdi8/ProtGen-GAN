# Environment Migration — Python 3.8 / PyTorch 2.4.1 / CUDA 11.8 → Python 3.12 / PyTorch 2.5.1 / CUDA 12.1

## Why

Python 3.8 reached end-of-life (Oct 2024). NumPy 1.24, SciPy 1.10, and Pillow 9 have no Python 3.12 builds.
The PyG C-extensions were compiled against `pt24cu118` and break under PyTorch 2.5.1.

## New environment file

`protgen-gan-env-v2.yml` — replaces `Conda-Environment-for-ProtGEN_mn5.yml`.

Key version floor changes enforced in the new file:

| Package | Old | New floor | Why |
|---------|-----|-----------|-----|
| Python | 3.8.19 | 3.12 | EOL |
| PyTorch | 2.4.1+cu118 | 2.5.1+cu121 | CUDA 12.1 target |
| torchvision | 0.19.1 | 0.20.1 | matches pt2.5.1 |
| torchaudio | 2.4.1 | 2.5.1 | matches pt2.5.1 |
| numpy | 1.24.4 | ≥1.26 | first release with Python 3.12 wheels |
| scipy | 1.10.1 | ≥1.11 | first release with Python 3.12 wheels |
| pillow | 9.4.0 | ≥10.0 | first release with Python 3.12 wheels |

## Code changes made during migration

### 1. `val_metrics.py:273` — `torch.load` weights_only (applied manually)

```python
# Before — FutureWarning in PyTorch 2.5, breaks in 2.6:
checkpoint = torch.load(checkpoint_path, map_location=device)

# After:
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
```

ProteinMPNN checkpoints contain plain Python dicts, not just tensors, so `weights_only=False`
is intentional. Without it, PyTorch 2.6+ will default to `True` and refuse to unpickle the checkpoint.

### 2. `generate.py` — deprecated `typing` imports removed

```python
# Removed:
from typing import List, Tuple

# All annotations updated to native generics (Python 3.9+):
# List[int]  →  list[int]
# List[str]  →  list[str]
# Tuple[X,Y] →  tuple[X, Y]
```

Affected function signatures: `load_length_distribution`, `load_seed_sequences`,
`sample_sequence_length`, `to_token_ids`, `resolve_model_path`,
`build_fully_masked_batch`, `build_seeded_batch`,
`generate_full_mode_sequences`, `generate_seeded_mode_sequences`.

## Build instructions

```bash
# 1. Create env (on Anzu — needs internet)
conda env create -f protgen-gan-env-v2.yml

# 2. Activate
conda activate protgen-gan-env

# 3. Install PyG C-extensions (must match pt2.5.1+cu121 ABI)
pip install pyg-lib torch-scatter torch-sparse torch-cluster \
            torch-spline-conv torch-geometric \
            -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# 4. Smoke test
python -c "import torch; import progres; print('PyG+progres OK')"
python -c "import mmtf; print('mmtf OK')"
python -c "from val_metrics import calculate_pairwise_tm_score; print('val_metrics OK')"
```

## MN5 deployment (air-gapped)

MN5 has no internet. Build the env on Anzu and ship it as a tarball:

```bash
# On Anzu
conda install conda-pack          # once, in base env
conda pack -n protgen-gan-env -o protgen-gan-env.tar.gz

scp protgen-gan-env.tar.gz ufuk@mn5:~/envs/

# On MN5
mkdir -p ~/envs/protgen-gan-env
tar -xzf ~/envs/protgen-gan-env.tar.gz -C ~/envs/protgen-gan-env
source ~/envs/protgen-gan-env/bin/activate
conda-unpack   # rewrites hardcoded Anzu paths
```

## VSCode note

Pylance infers Python version from the active interpreter. After switching the VSCode
Python interpreter to the new 3.12 env, any `list[int]` / `tuple[X,Y]` annotation errors
from the migration will disappear.
