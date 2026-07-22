# Archive

## Week of 2026-06-29
Audited ProtGen GAN (13 critical issues: transformer skip, non-diff generator, diversity gaps, training dupes, hardcoded paths); rejected conda→uv; added .claude/ hooks (.env/.yml/slurm); documented soft-embed train/eval mismatch & updated memory.

## Week of 2026-06-23
Fixed wandb auth & training dtype/SDPA bugs; validated baseline (plddt=0.66). Implemented 3-stage grad-flow: Stage 0 (instrumentation), Stage 1 (soft embeddings), Stage 2 (KL anchor + grad clipping). Fixed NaN (KL log-near-zero) & OOM (3 concurrent ProtBERTs). Identified KL design flaw (unmasked seqs OOD for MLM)—resolved via 50% masking. Memory optimization: fp16 ref_protbert + ESMFold CPU + seq len cap (500→350) freed 3.6GB. Added --max_train_seqs/--iteration_fill_rate flags. Training converged, validation ready.

## Week of 2026-06-16
Dev infrastructure & training stabilization. Resolved PyG C-ext, RTK setup, wandb shadowing. Fixed training bugs (pLDDT, masking, NaN, temp). Added QoL (unique_ratio, opt ckpt, seq-cache). PyTorch 2.5.1 compat (SDPA, dtype). Implemented ruff/mn5 hooks. Env audit (Py→3.12, PyT→2.5.1). Updated ENV_MIGRATION, CLAUDE, config.

## Week of 2026-05-12
PyTorch 2.6+ compat: torch.load(weights_only=False) in val_metrics.py, protgen-gan-env-v2.yml with updated deps (numpy≥1.26, scipy≥1.11, pillow≥10, cuda=12.1, PyG pt25cu121). Assessed conda→uv migration for ProtGEN HPC—rejected due to CUDA/C++ deps, retained conda with per-machine .yml validation. Smoke tests pass, codebase clean.
```