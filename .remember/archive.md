# Archive

## Week of 2026-06-29
Audited ProtGen GAN (13 critical issues: transformer skip, non-diff generator, diversity gaps, training dupes, hardcoded paths); rejected conda→uv; added .claude/ hooks (.env/.yml/slurm); documented soft-embed train/eval mismatch & updated memory.

## Week of 2026-06-16
Dev infrastructure & training stabilization. Resolved PyG C-ext, RTK setup, wandb shadowing. Fixed training bugs (pLDDT, masking, NaN, temp). Added QoL (unique_ratio, opt ckpt, seq-cache). PyTorch 2.5.1 compat (SDPA, dtype). Implemented ruff/mn5 hooks. Env audit (Py→3.12, PyT→2.5.1). Updated ENV_MIGRATION, CLAUDE, config.

## Week of 2026-05-12
PyTorch 2.6+ compat: torch.load(weights_only=False) in val_metrics.py, protgen-gan-env-v2.yml with updated deps (numpy≥1.26, scipy≥1.11, pillow≥10, cuda=12.1, PyG pt25cu121). Assessed conda→uv migration for ProtGEN HPC—rejected due to CUDA/C++ deps, retained conda with per-machine .yml validation. Smoke tests pass, codebase clean.
```