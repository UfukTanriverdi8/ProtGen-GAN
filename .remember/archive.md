# Archive

## Week of 2026-07-27
Fixed multi-machine git sync (MN5↔Anzu↔GitHub) and RNG seeding (added --seed flag, default 89). Phase 2 λ_kl-sweep finalized params (0.005, 0.05); added wandb-run-reviewer subagent, rewrote slurm-job skill. Diagnosed quota failure: 10p_train.py storing full ProtBERT+opt-state per epoch (~8GB)—freed 4TB+. Root-caused wandb config persistence gap and critic saturation (constant-output collapse); identified missing held-out eval in protgen validation. Flipped λ_kl rec to 0.05.

## Week of 2026-07-20
Soft-embed gradient debugging validated (gen_grad_norm>0, KL loss 0–4 vs 89–1236 pre-fix). Implemented λ_kl sweep Phase 1 (n_critic=4, temp=1.0); Phase 1 completed with finalized Phase 2 params (λ_kl: 0.005, 0.05). Merged mn5/dev commits; added --wandb_tags flag; rewrote slurm-job/pre-submit skills, retired wandb-sync skill. Docs audit: fixed stale entries (GRADIENT_FIX_EXPLAINED, ENV_MIGRATION, lambda-kl-sweep).

## Week of 2026-06-29
Audited ProtGen GAN (13 critical issues: transformer skip, non-diff generator, diversity gaps, training dupes, hardcoded paths); rejected conda→uv; added .claude/ hooks (.env/.yml/slurm); documented soft-embed train/eval mismatch & updated memory.

## Week of 2026-06-23
Fixed wandb auth & training dtype/SDPA bugs; validated baseline (plddt=0.66). Implemented 3-stage grad-flow: Stage 0 (instrumentation), Stage 1 (soft embeddings), Stage 2 (KL anchor + grad clipping). Fixed NaN (KL log-near-zero) & OOM (3 concurrent ProtBERTs). Identified KL design flaw (unmasked seqs OOD for MLM)—resolved via 50% masking. Memory optimization: fp16 ref_protbert + ESMFold CPU + seq len cap (500→350) freed 3.6GB. Added --max_train_seqs/--iteration_fill_rate flags. Training converged, validation ready.

## Week of 2026-05-12
PyTorch 2.6+ compat: torch.load(weights_only=False) in val_metrics.py, protgen-gan-env-v2.yml with updated deps (numpy≥1.26, scipy≥1.11, pillow≥10, cuda=12.1, PyG pt25cu121). Assessed conda→uv migration for ProtGEN HPC—rejected due to CUDA/C++ deps, retained conda with per-machine .yml validation. Smoke tests pass, codebase clean.