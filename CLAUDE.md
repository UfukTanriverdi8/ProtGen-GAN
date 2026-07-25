# ProtGen — GAN Repo Context for Claude Code

## Commands

### Environment Setup
```bash
# Required on every machine before running any script.
# MN5:  add to your SLURM job script
# Anzu: add to ~/.bashrc
export SOURCE_DIR=/gpfs/projects/etur29/ufuk                   # MN5 example
export SOURCE_DIR=/media/ubuntu/8TB/ufuk/protgen-gan/models     # Anzu example
# config.py resolves PROTBERT, ESMFold, and checkpoint paths from SOURCE_DIR.
# MN5 hostnames are auto-detected as a fallback if SOURCE_DIR is not set.
```

### Training (MN5 — submit via SLURM)
```bash
sbatch slurms/long_10p_run.sh        # Seeded mode extended run
sbatch slurms/lr_full_run.sh         # Blind mode LR grid search

# Direct invocation (Anzu / local debugging)
python 10p_train.py --run_name debug_run --n_epochs 2 --batch_size 4
python fully_masked_train.py --run_name debug_run --n_epochs 2 --batch_size 4

# Fast test run (10k seqs, no ESMFold load, clean adversarial dynamics)
python 10p_train.py --run_name test-run --n_epochs 3 --batch_size 4 --n_critic 4 \
  --num_eval_sequences 20 --max_train_seqs 10000 --lambda_kl 0.0
# --num_eval_sequences 0  → skips ESMFold load entirely (~2.8GB VRAM saved); losses still log.
# --max_train_seqs N      → caps dataset to N sequences for faster epoch cycles.
# --iteration_fill_rate F → fraction filled per generation step (default 0.1 = 10 steps).
# --lambda_kl 0.0         → disables KL anchor; use for clean adversarial-only test runs.
```

### Generation (MN5)
```bash
# From a trained GAN checkpoint
python generate.py --ckpt_id <run_name>/epoch_<N> --num_full 5000 --num_seeded 5000

# From the fine-tuned ProtBERT baseline (no GAN training)
python generate.py --ckpt_id finetuned_protbert --num_full 5000 --num_seeded 5000
```

### Evaluation
```bash
sbatch slurms/mass_eval.sh                       # Batch structural eval (120k+ seqs)
python eval_sequences/check_duplicates.py        # Deduplication stats
python eval_sequences/prefilter_sequences.py     # Remove length >350, X tokens
python eval_sequences/build_csv_from_txt.py      # Aggregate generation outputs into CSV
```

---

## Repo Scope

This repository contains the GAN training, generation, and evaluation pipeline for ProtGen.
It does **not** yet include the ProtBERT fine-tuning code or the AlphaFold3 analysis pipeline —
those will be integrated into this repo later.

---

## Project Overview

**ProtGen** is the working codename for this project — the final published name will be
something more original, TBD.

ProtGen is a de novo protein design system developed at Hacettepe University BioDataSciLab
(Prof. Tunca Doğan's group). The goal is to generate functional artificial protein sequences
that mimic the DNMT family (DNA methyltransferases) — enzymes that catalyse cytosine
methylation, a key epigenetic regulator.

The generative backbone is a fine-tuned ProtBERT model integrated into a GAN-like architecture.
ProtBERT was fine-tuned in two successive stages before being used here:

1. Broad pre-training on ~600k EC 2.1.1 transferase sequences
2. Specialization on ~50k DNMT sequences (EC 2.1.1.37, IPR001525 domain)

The final fine-tuned checkpoint (dynamic masking, epoch 300 of stage 2) is the backbone
for both the generator and the critic in this repo.

## Similar Paper by the same lab: DrugGEN

DrugGEN is a GAN-based de novo drug design system from the same lab (Tunca Doğan's group, published in Nature Machine Intelligence 2025). It generates small drug-like molecules targeting specific proteins, representing molecules as graphs and using graph transformer-based generator and discriminator. Unlike ProtGen which iteratively fills masked protein sequences, DrugGEN generates full molecular graphs in one shot. Despite the architectural differences, several lessons transfer directly. There can be lessons learned from DrugGEN in this project so some of them are listed here:

**Training stability:**

- Uses WGAN-GP (gradient penalty λ=10) — ProtGen's GP had two bugs: it bypassed the full transformer and ignored padding masks. Both were fixed in `loss.py` (commit 8ebec32), and the calling code in both training scripts has since been updated with the correct signature.
- AdamW optimizer with lr=1e-5 for both G and D
- Trains discriminator first before generator joins — similar to ProtGen's first-epoch freezing rationale
- Used early stopping based on validity/novelty metrics to catch mode collapse before it ran for months undetected

**Evaluation during training:**

- Tracks uniqueness and novelty throughout training, not just loss — a lightweight uniqueness check on small generated batches during training would have caught ProtGen's mode collapse much earlier
- Multiple independent metrics rather than relying on any single one

**Architecture:**

- Transformer depth of 1 was optimal — higher depth hurt convergence
- GAN training with deeper/larger backbones tends to be less stable, not more

**Downstream pipeline:**

- Multi-stage filtering: docking → DL-based bioactivity prediction → MD simulation → wet lab. No single metric was trusted alone. This validates ProtGen's finding that progres/scAccuracy alone are insufficient.

---

## Architecture

### Generator
- A copy of the fine-tuned ProtBERT with no modifications.
- Takes a partially or fully masked sequence as input.
- At each iteration: runs ProtBERT forward, applies softmax, fills the top 10% most confident
  masked positions, re-masks the rest. Repeats until the sequence is fully filled.
- Token selection at each step is controlled by `temperature` in `models.py`.

### Critic
- A second copy of the fine-tuned ProtBERT with an added feedforward classification head (MLP).
- Trained to distinguish real DNMT sequences from generated ones.
- Uses WGAN-style loss.

### First-Epoch Freezing
- During epoch 1, both ProtBERT instances are frozen. Only the classification head trains.
- This warms up the head before the adversarial game starts, since the ProtBERT weights are
  already specialized but the head starts from random initialization.

---

## Training Modes

### Seeded Mode (`10p_train.py`)
- Dataset is split: half for the generator, half for the critic.
- Generator inputs: real sequences with 90% of tokens masked, leaving 10% as a seed.
- The seed provides context that guides generation from a partially known sequence.
- File naming convention: runs include `10p` or `nc8` (n_critic=8) in the name.

### Blind Mode (`fully_masked_train.py`)
- Critic trains on the entire dataset.
- Generator inputs: fully masked sequences of lengths sampled from the dataset distribution.
- No seed — the generator must produce a sequence from scratch.
- File naming convention: runs include `full` in the name.

---

## ✅ FIXED — Temperature/Argmax Bug (`models.py`)

**Fixed in:** `models.py` → `Generator.generate()`

The original code applied temperature to logits then called `.max()` (argmax), making
temperature completely ineffective. Argmax always picks the same token regardless of the
distribution shape. Fix: replaced `.max()` with `torch.multinomial()` so temperature
actually controls sampling diversity.

**Consequence of the unfixed version:** blind mode produced ~300 unique sequences out of
10,000 (one per sampled length). GAN training saw zero diverse fakes for ~60-70 runs.
Seeded mode was accidentally functional only because different seeds produced different logits.

---

## ✅ FIXED — Generator Gradient Flow Fix (All Stages Done)

**Affects:** Every training run ever. Not a one-line fix — required architectural change.
**Status (2026-06-26):** All 3 stages implemented. Generator now receives adversarial
gradient for the first time. Read `docs/GENERATOR_GRADIENT_FIX.md` before touching
`models.py`, `loss.py`, or either training script for this issue.

### The Problem

The generator's `generate()` method returns discrete `torch.long` token IDs. PyTorch
integer tensors carry no gradient. When those token IDs are passed to the critic:

```python
# In both training scripts (generator update step):
fake_data = generate_fakes_for_batch(...)   # returns torch.long token IDs
fake_scores = critic(fake_data, ...)        # embedding lookup on integers
g_loss = generator_loss(fake_scores)
g_loss.backward()                           # gradient dies at the integer boundary
gen_optimizer.step()                        # applies zero gradient to generator
```

The critic's embedding lookup is not differentiable w.r.t. integer input IDs — they're
indices into a lookup table, not continuous values. `g_loss.backward()` propagates
gradients through the critic's own parameters, but cannot cross back through the discrete
sampling step into the generator's ProtBERT weights.

**Result:** The generator backbone (ProtBERT) has never been updated by adversarial signal.
The only thing that changes during "generator updates" is weight decay from AdamW gradually
eroding the fine-tuned weights. The GAN has never functioned as a GAN — it has been using
the fine-tuned ProtBERT as-is for generation, with the critic learning to classify against
a static generator.

This is true both with the old argmax and with the new `torch.multinomial()` — both produce
discrete integers, neither is differentiable.

### Decided fix (see `docs/GENERATOR_GRADIENT_FIX.md` for full justification)

Soft embedding pass-through, **not** Gumbel-Softmax and **not** REINFORCE for the
adversarial term — the critic is differentiable, so a continuous relaxation is the right
family. Decided approach in full:

**Stage 0 ✅** — `gen_grad_norm` logged after `g_loss.backward()` in both training scripts.
Confirmed ≈0 before fix (proves the bug). Should be > 0 after Stage 1.

**Stage 1 ✅** — Soft embeddings + embed-the-real + GP in embedding space:
- `compute_soft_embeds()` in `models.py`: re-masks 50% of the completed sequence (excluding
  special tokens), runs one generator forward pass on that masked input →
  `softmax(logits/T) @ critic_word_weight`, then blends soft embeds at the remasked positions
  with hard original-token embeds elsewhere → full embeds via `inputs_embeds`. Gradient reaches
  the generator only through the remasked slots. The iterative fill loop is NOT in the gradient
  graph (K=1 truncated backprop, natural).
- Both real and fake embedded via `critic.protbert.bert.embeddings()` before critic update
  so critic cannot distinguish real/fake by embedding sparsity.
- `compute_gradient_penalty()` now accepts pre-computed `[B,L,H]` embedding tensors.

**Stage 2 ✅** — KL anchor against frozen reference ProtBERT:
- `compute_kl_anchor()` in `models.py`: `KL(generator || frozen_ref)` at same temperature,
  computed only at the remasked positions — both models see the identical masked input.
- `ref_protbert` loaded from `PROTBERT_PATH`, frozen, eval — never updated.
- `g_loss = -critic(soft_embeds).mean() + lambda_kl * KL(gen || ref)` (default `lambda_kl=0.01`).
- `kl_loss` logged to wandb. Tune `--lambda_kl` if generator drifts too fast or too slow.

**Next: validate in seeded mode.** Watch `gen_grad_norm` (> 0), `kl_loss` (stable, not
exploding), and `unique_ratio` (not collapsing). Blind mode has mode-collapse risk —
test seeded first. Full validation criteria in `docs/GENERATOR_GRADIENT_FIX.md`.

**✅ FIXED — KL anchor / soft embeds now computed on a re-masked input (2026-06-27):**
Previously `compute_soft_embeds` (and thus `compute_kl_anchor`) received `fake_data` after
the iterative fill loop — fully revealed, zero [MASK] tokens. ProtBERT is MLM-trained and
had never seen fully-revealed input; logits were uncalibrated in that regime, and `kl_loss`
oscillated 89–1236 throughout training even with `gen_probs.clamp(min=1e-8)` and
`clip_grad_norm_(max_norm=1.0)`. Fix (commit `c18c89a`): `compute_soft_embeds` now re-masks
50% of the completed sequence and runs the generator on that, so ProtBERT stays
in-distribution; `compute_kl_anchor` evaluates the frozen reference on the same masked input
and scores KL only at the remasked positions. Applied to both training scripts. `--lambda_kl`
can now be left at its default; `--lambda_kl 0.0` remains available for clean adversarial-only
runs.

**✅ VALIDATED (2026-06-27, runs `xwres3cz` and `ydsjq9f4`, both 3-epoch seeded-mode, n_critic=4):**
`gen_grad_norm > 0` from epoch 2 on in both runs, bounded (not exploding like pre-fix's
100k+ spikes). `kl_loss` (lambda_kl=0.01 run) now oscillates in a bounded 0–4 range instead
of the pre-fix 89–1236 — fix confirmed. `unique_ratio` held at 1.0 in both runs. The
lambda_kl=0 run additionally shows the KL anchor is load-bearing for sequence *quality*:
without it, `plddt`/`scAccuracy`/`progres`/`pairwise_tm` all collapsed by epoch 3 even
though uniqueness never dropped — diverse but non-protein-like. Full numbers in
`docs/GENERATOR_GRADIENT_FIX.md` Stage 3. These runs were never analyzed in a Claude Code
session at the time; recovered from wandb run history on 2026-07-22.

`tests/check_kl_identity.py` independently verifies `compute_kl_anchor`'s plumbing (not
just training-curve behavior) via `KL(P‖P)=0` + a perturbed-weights negative control. Run
it before trusting future changes to `compute_soft_embeds`/`compute_kl_anchor`.

---

## ✅ FIXED — `compute_gradient_penalty` (`loss.py` + both training scripts)

Now accepts pre-computed `real_embeds` and `fake_embeds` (`[B,L,H]` float tensors).
Callers embed real and fake before calling. Signature:

```python
gp = compute_gradient_penalty(
    critic, real_embeds, fake_embeds_hard,
    attn_mask_real,
    fake_mask,
    device
)
```

---

## ✅ FIXED — `generate.py` relied on `Generator`'s hardcoded `mask_token_id` default

**File:** `generate.py:372`
**Was:** Never passed `mask_token_id`, silently using the class default (`4`) — only
correct by coincidence with this checkpoint's vocab. A different checkpoint's tokenizer
would silently mask the wrong token with no error.
**Fixed (`718e446`):** Passes `mask_token_id=tokenizer.mask_token_id` explicitly, like
`10p_train.py`/`fully_masked_train.py` already did.

---

## ✅ BUGS FOUND — Fixed (2026-06-19)

These were identified by auditing the codebase after the temperature fix.

---

### ✅ BUG 2 — `calculate_plddt_scores_and_save_pdb` tuple unpacking

**Files:** `10p_train.py:231`, `fully_masked_train.py:160`
**Was:** CRITICAL — silently logged a tuple to wandb instead of a float; all pLDDT
metrics in every training run were garbage.
**Fixed:** `avg_plddt_score, _ = calculate_plddt_scores_and_save_pdb(...)`

---

### ✅ BUG 3 — `generate_fake_sequences` dead temperature variable

**File:** `val_metrics.py:119–131`
**Was:** Significant — evaluation sequences always used temperature=1.0; the random
temperature variation was computed but silently discarded via `fixed_temp`.
**Fixed:** Removed `fixed_temp`, now passes `random_temp` to `generator.generate()`.

---

### ✅ BUG 4 — `fully_masked_train.py` attention mask excluded [MASK] tokens

**File:** `fully_masked_train.py:222`
**Was:** Significant — mask tokens were invisible in attention, degrading blind mode
generation quality. Used `mask_token_id` instead of `pad_token_id`.
**Fixed:** `updated_attention_mask = (final_input_ids != tokenizer.pad_token_id).long()`

---

### ✅ BUG 5 — NaN guard in `calculate_plddt_scores_and_save_pdb` overwritten

**File:** `val_metrics.py:200`
**Was:** Moderate — NaN-safe average was immediately overwritten by unsafe `sum/len`.
**Fixed:** Deleted the unsafe second `if` block.

---

### ✅ BUG 6 — `generate_fake_batch` defaulted `debug=True`

**File:** `fully_masked_train.py:197`
**Was:** QoL — flooded SLURM logs with per-position mask counts on every batch.
**Fixed:** Changed default to `debug=False`.

---

## ⚠️ QUALITY OF LIFE — Missing Features Worth Adding

### ✅ QoL 1 — Uniqueness metric added to in-training evaluation (2026-06-19)

**Files:** `10p_train.py`, `fully_masked_train.py` → `run_evaluation()`
**Fixed:** `unique_ratio` computed and logged to wandb after sequence generation.
Cheapest mode-collapse tripwire — would have caught the argmax bug on the first run.

### ✅ QoL 2 — Optimizer state now saved in checkpoints (2026-06-19)

**Files:** `10p_train.py`, `fully_masked_train.py` → checkpoint saving block
**Fixed:** `gen_optimizer.pth` and `critic_optimizer.pth` saved alongside model weights.
Load logic to be added when resume script is built (QoL 5).

### ✅ QoL 3 — `sample_sequence_length` cached via `lru_cache` (2026-06-19)

**File:** `val_metrics.py:49–53`
**Fixed:** File read extracted into `_load_sequence_lengths()` with `@functools.lru_cache`.
52k-line dataset file read once, cached for all subsequent calls.

### QoL 4 — `from val_metrics import *` loads the entire evaluation stack at training startup

**Files:** `10p_train.py:10`, `fully_masked_train.py:10`

Both training scripts wildcard-import `val_metrics`, which at module load time imports
ESMFold, ProteinMPNN, BioPython, progres, tmtools, and requests. If any dependency is
missing or broken, training crashes before a single data batch is processed. Consider
importing only what's needed, or wrapping heavy imports inside the functions that use them.

### QoL 5 — Both training scripts will be merged ✅ DECIDED

`10p_train.py` and `fully_masked_train.py` are ~80% duplicate. Decision made (2026-05-14):
merge into a single script with a `--mode seeded|blind` flag before next training round.
Until merged, any bug fix must be applied to both files manually.

---

## File Structure

```
gan/
├── # CORE TRAINING
├── 10p_train.py               Seeded mode GAN training loop (10% seed, 90% masked)
├── fully_masked_train.py      Blind mode GAN training loop (fully masked sequences)
├── generate.py                Inference script — generates sequences from a checkpoint
│                              (supports both full/seeded modes, 10k sequences per run)
│
├── # MODEL & LOSS
├── models.py                  Generator and Critic classes wrapping fine-tuned ProtBERT
├── loss.py                    Wasserstein loss + gradient penalty for WGAN training
├── config.py                  Path resolution: reads SOURCE_DIR env var, falls back to
│                              MN5 hostname detection. Exports PROTBERT_PATH, ESMFOLD_PATH,
│                              CHECKPOINT_DIR, PROTBERT_BASE.
│
├── # DATA
├── dataset.py                 PyTorch Dataset/DataLoader: tokenisation, batching,
│                              gen/critic split for seeded mode and fully-masked mode
└── data/
    ├── file_formatter.py      Splits raw sequence CSV into train/val/gen/critic splits
    ├── test_data.py           Sanity checks: uniqueness, length distribution
    ├── dnmt_unformatted.txt   Raw sequences (one per line)
    ├── dnmt_full.txt          Space-separated sequences (full dataset)
    ├── dnmt_gen.txt           ~50% split for generator training
    └── dnmt_critic.txt        ~50% split for critic training

├── # EVALUATION
├── val_metrics.py             All evaluation metric functions:
│                              pLDDT (ESMFold), TM-score, scAccuracy (ProteinMPNN),
│                              PROGRES, pairwise TM-score diversity, seq_similarity
├── run_mass_eval.py           Batch evaluation orchestrator (processes 120k+ sequences)
├── test_model_and_metrics.py  Dev harness for testing metric pipeline
├── tests/
│   └── check_kl_identity.py   Verifies compute_kl_anchor via KL(P‖P)=0 identity + negative control
└── eval_sequences/
    ├── build_csv_from_txt.py  Aggregates raw .txt generation outputs into a CSV
    │                          (parses run name / mode / epoch from filename)
    ├── check_duplicates.py    Deduplication stats and most-common-sequence frequencies
    ├── prefilter_sequences.py Filters length > 350, removes X tokens, drops nulls
    ├── merge_csv.py           Merges multiple CSV files into a master dataset
    ├── check_mass_eval_results.py  Reports on mass eval completion status
    └── 120k_eval/
        ├── 120k_eval_seqs.csv              Raw aggregated 120k sequences
        ├── 120k_eval_seqs_prefiltered.csv  After length/X-token filtering
        └── 120k_eval_seqs_final.csv        After full metric evaluation

├── # SLURM JOB SCRIPTS (MN5)
└── slurms/
    ├── generate_samples.sh        Array job: generate 10k sequences from 6 checkpoints
    │                              (full + seeded in parallel per checkpoint)
    ├── mass_eval.sh               Array job: parallel metric evaluation of generated seqs
    ├── long_10p_run.sh            Extended training run for seeded mode
    ├── lr_10p_run.sh              LR grid search for seeded mode (25 combinations)
    ├── lr_full_run.sh             LR grid search for blind mode
    ├── lr_best_runs.sh            Retraining with best LRs from sweep
    ├── n_critic_10p_run.sh        n_critic grid search (seeded mode)
    ├── n_critic_full_run.sh       n_critic grid search (blind mode)
    ├── env_smoke_test.sh          Quick sanity check that the packed conda env loads correctly
    └── sweeps/                    Hyperparameter sweep array jobs (split out from the flat
                                   layout above); see docs/RUN_DOCUMENTATION.md for how
                                   results are recorded
        └── lambda_kl_sweep_p1.sh  Phase 1 screening: lambda_kl grid, seeded mode, n_critic=4

├── # VALIDATION / VISUALISATION
└── validation/
    ├── visualize.ipynb        Jupyter notebook for plotting metrics and results
    ├── pdbs/reference/        Reference DNMT3A PDB structure
    └── test_pdb/              PDB output files from metric computation

├── # OUTPUTS
└── outputs/                   SLURM stdout/stderr logs from training and eval jobs

├── # CONFIG & DOCS
├── CLAUDE.md                  This file
├── README.md                  High-level project overview
├── envs/
│   ├── protgen-gan-env-v2.yml   The only valid/active env spec (Python 3.12, PyTorch 2.5.1,
│   │                            CUDA 12.1) — identical on Anzu and MN5, see Infrastructure below
│   ├── anzu-env-export.yml      Backup snapshot taken during the env migration — not for use
│   └── mn5-env-export.yml       Backup snapshot taken during the env migration — not for use
├── bfg-1.15.0.jar             BFG repo cleaner (git history cleanup utility)
└── docs/
    ├── RUN_DOCUMENTATION.md   Generic convention for docs/runs/ + docs/sweeps/ (below)
    ├── runs/                  One markdown file per individual training run
    │                          (config, final metrics, per-epoch trend, verdict)
    └── sweeps/                One markdown file per sweep (design ref, run table, winner
                               selection reasoning, conclusion)
```

> Fine-tuning code and AF3 analysis scripts will be added to this repo later.

---

## Evaluation Metrics (used in `eval.py`)

All structural analysis uses ESMFold to predict structure from sequence first.

| Metric | Description |
|--------|-------------|
| `pLDDT` | ESMFold structure prediction confidence score (0–1). Filter threshold: >= 0.8 |
| `scAccuracy` | Self-consistency: send sequence → ESMFold → ProteinMPNN → alignment score between original and re-predicted sequence |
| `progres` | Structural similarity to a reference DNMT protein. Previously used as a filter (>= 0.9) but **shown to be a poor predictor of DNA-SAM binding** — do not use as a hard filter |
| `pairwise TM-score` | Diversity metric: TM-score between each generated sequence pair |
| `seq_similarity` | Sequence similarity to reference DNMT. Filter threshold: <= 0.35 |

### Pre-AF3 Filter (current best practice, based on evaluation findings)

```python
(df['length'] <= 300) &
(df['plddt'] >= 0.8) &
(df['seq_similarity'] <= 0.35)
# Note: progres and scaccuracy intentionally excluded — see findings below
```

Also applied before custom metrics: remove sequences with length > 350 and sequences
containing the `X` amino acid token (unknown residue — problematic for ESMFold/AF3).

---

## Key Evaluation Findings (from pre-fix runs)

These findings are based on seeded mode results only (blind mode was collapsed due to the bug):

- **progres and scAccuracy are poor predictors of DNA-SAM binding.** The best 4 surviving
  sequences (5.27–6.68 Å) came from `batch1_relaxed` where both filters were removed.
  Sequences with high progres scores did not outperform those without.
- **All 9 unique survivors came from seeded mode.** Blind mode contributed zero meaningful
  sequences — consistent with total mode collapse.
- **Best candidate:** sequence `2508_len279` from `full_nc8_lrgen_5e-5_lrcrit_5e-4` (seeded),
  RMSD 5.272 Å to reference DNA-SAM binding site. Structurally validated by a chemistry
  professor. Secondary structure similar to DNMT-3a with hydrophilic interactions for SAM binding.
- **30 sequences returned AF3 errors** and remain unevaluated — potential candidates.
- **scAccuracy >= 0.3 filter in 120k batch excluded the second-best candidate** (5.425 Å RMSD).
  Don't use scAccuracy as a hard filter going forward.

---

## Hyperparameter Search History

After extensive search (n_critic values: 1, 2, 4, 8, 12; learning rates for both gen and critic),
best performing configurations identified were:

**Seeded Mode best LR combinations (lr_gen, lr_crit):**
- `5e-6, 5e-5`
- `5e-5, 5e-4`

**Blind Mode best LR combinations:**
- `1e-5, 5e-5`
- `5e-5, 5e-4`

Current standard: `n_critic = 8`, first epoch frozen.

> Note: all of these runs were conducted under the argmax bug. Hyperparameter behaviour
> may change meaningfully after the `torch.multinomial()` fix. Re-validation recommended.
> It may change again, possibly more significantly, once the gradient-flow fix in
> `docs/GENERATOR_GRADIENT_FIX.md` is implemented and the generator actually starts
> receiving adversarial signal for the first time.

---

## Infrastructure

### Key Documents
- `docs/INFRASTRUCTURE.md` — hardware specs for each machine (Anzu GPU inventory/VRAM ceiling, RAM/CPU, OS). Read before suggesting batch sizes or configs for a specific machine, especially Anzu where VRAM varies 24-48GB across its 7 GPUs.
- `docs/HISTORY.md` — full project narrative: every phase, architectural decision, bug discovery, and current state. Read before suggesting experiments or evaluating what's been tried.
- `docs/GIT_WORKFLOW.md` — complete two-remote git workflow and wandb offline sync. Includes agent-specific notes at the bottom.
- `docs/GENERATOR_GRADIENT_FIX.md` — full research synthesis and staged implementation plan for the non-differentiable-generator architectural issue (above). Read before touching `models.py`, `loss.py`, or either training script in relation to that issue.
- `docs/GRADIENT_FIX_EXPLAINED.md` — conceptual companion to the above; explains the gradient problem, soft embeddings, KL anchor, and related concepts from first principles. No implementation details — read for understanding.
- `docs/ENV_MIGRATION.md` — rationale and code changes for the Python 3.8/PyTorch 2.4.1 → Python 3.12/PyTorch 2.5.1 environment migration (`envs/protgen-gan-env-v2.yml`). Read before touching env files or diagnosing version-related errors.
- `docs/RUN_DOCUMENTATION.md` — generic convention for recording individual runs (`docs/runs/`) and sweeps (`docs/sweeps/`) in plain markdown, so results can be revisited without digging through wandb history. Read before starting any new training run or sweep, and use it to write up results once a run/sweep finishes.

### Claude Code Automation (`.claude/`)
- **Hook: file protection** — blocks edits to `.env` and `protgen-gan-env-v2.yml` (path-substring match, so it still applies now that the file lives under `envs/`)
- **Hook: ruff auto-lint** — runs `ruff check` on every `.py` file after Edit/Write
- **Hook: mn5 push guard** — requires confirmation for `git push mn5` or force-push
- **Skill: `slurm-job`** — generates MN5 SLURM scripts from run parameters
- **Skill: `bug-fix-checklist`** — Claude-only; greps for all known unfixed bugs before touching training/eval files
- **Skill: `pre-submit`** — validates codebase state (bugs, env, wandb) before SLURM submission
- **Subagent: `wandb-run-reviewer`** — documents a single synced wandb run into `docs/runs/` per `RUN_DOCUMENTATION.md`'s template (full per-epoch history via wandb MCP, not just final values). Does not rank/compare runs — that's a separate synthesis step done inline, not delegated.

| Environment | Purpose |
|-------------|---------|
| **Anzu** | Hacettepe BioDataSciLab GPU server (Ubuntu). Used for eval runs, debugging, smaller experiments. SSH access. |
| **MareNostrum5 (MN5)** | BSC supercomputer, thousands of H100s. Used for large GAN training runs and AF3 evaluation. **No internet access** — files transferred via SCP (upload) and SFTP (download). |

### Environment (conda-pack migration, complete as of 2026-07-17)
The `protgen-gan` conda env (`envs/protgen-gan-env-v2.yml`) is now identical on Anzu and MN5, packed
with `conda-pack` and transferred via SCP since MN5 has no internet for a normal conda install.

- MN5 install path: `/gpfs/projects/etur29/ufuk/envs/protgen-gan`
- Activate with the packed env's own script, **not** `conda activate`:
  `source /gpfs/projects/etur29/ufuk/envs/protgen-gan/bin/activate`
- Never mix `conda activate`/`deactivate` with this env on MN5 — the two activation mechanisms
  partially clobber each other's env vars (`CONDA_PREFIX`, `PATH`) and deactivation silently
  fails. Use a fresh shell instead of trying to deactivate.
- SLURM scripts already updated to use the packed env; see `slurms/*.sh`.

**`progres` database files:** `progres` auto-downloads its trained model + pre-embedded databases
(~830MB) from Zenodo on first use — this fails on MN5 (no internet). These files were manually
copied from Anzu (where they already existed as a side effect of local eval runs) via `rsync`
into MN5's `PROGRES_DATA_DIR` (`/gpfs/projects/etur29/ufuk/progres/`), preserving the
`trained_models/`, `databases/`, and `chainsaw/model_v3/` subdirectory structure plus each file's
`.pt.okay` marker (which tells `progres` to skip re-downloading). If this directory is ever lost
or `progres`'s `zenodo_record`/`database_subdir` version bumps, re-run the same rsync from Anzu
rather than trying to download directly on MN5.

### Git Workflow (Two-Remote Setup)
MN5 has no internet access, so it cannot push/pull directly to GitHub. The local machine
(or Anzu) acts as a mediator:

- `origin` → GitHub (remote)
- `mn5` → MN5 local repo (remote)

Typical flow: develop locally → push to `origin` (GitHub) and/or push to `mn5` directly.
To sync MN5 with GitHub, pull from `origin` locally then push to `mn5`, or vice versa.

**Commit messages:** never add a `Co-Authored-By: Claude` (or any Anthropic/Claude attribution)
trailer to commits in this repo, regardless of default tooling conventions.

### Secrets (`.env`)
Copy `.env.example` to `.env` and fill in real values (`WANDB_API_KEY`, `WANDB_MODE`). `.env` is
gitignored and never committed — it must exist independently on every machine that runs training
(Anzu, MN5), since `git push`/`pull` won't carry it. SLURM scripts load it via
`set -a; source "$SLURM_SUBMIT_DIR/.env"; set +a`, which auto-exports every var without needing
any parsing library.

### wandb Workflow on MN5
MN5 has no internet. wandb runs are logged offline, then synced from Anzu once transferred over.
Current flow (as of 2026-07-25), driven by two shell aliases on Anzu — not a Claude Code skill,
since the actual sync/transfer steps must be run by a human with MN5 SSH access:

1. **Stage on MN5:** once a run finishes, move its `wandb/offline-run-*/` directory into
   `wandb/departure/` before pulling — keeps in-progress runs untouched (never rsync a
   `.wandb` file mid-write) and gives explicit control over what gets pulled next.
2. **Pull run directories:** `get-mn5-wandb-runs` alias — rsyncs everything in
   `wandb/departure/` on MN5 to `/home/ufuk/protgen/mn5/arrivals/` on Anzu.
3. **Pull SLURM logs:** `get-mn5-logs` alias — rsyncs the whole `outputs/` folder from MN5 to
   this repo's `outputs/` on Anzu (git-tracked; a resubmitted job with the same
   `#SBATCH --output=` pattern will overwrite the old log file, so commit before resubmitting
   if you want the prior run's log preserved in history).
4. **Identify which folder is which run:** offline-run folder names
   (`offline-run-<timestamp>-<run_id>`) don't encode the human-readable `run_name` and the
   `run_id` isn't known until MN5 assigns it — but each array task's `.err` log (from step 3)
   contains a `wandb: Run data is saved locally in <path>` line, and the SLURM script's
   `kl_list`-style array already gives a static array-task-id → `run_name` mapping. Combining
   the two identifies every folder without needing to open `config.yaml` or guess.
5. **Sync to wandb cloud:** `wandb sync` each folder in `arrivals/` (a simple loop over
   `arrivals/offline-run-*/` — no dedicated tooling needed for this step).
6. **Document (optional):** once synced, the `wandb-run-reviewer` subagent (see Claude Code
   Automation below) can pull each run's full history via wandb MCP and write a
   `docs/runs/<run_name>_<run_id>.md` file per `RUN_DOCUMENTATION.md`'s convention.

### Evaluation Frequency
The evaluation frequency inside the training loop should be dynamic and proportional to
`n_critic`, since dataset distribution is split dynamically per epoch. Static every-250-batch
evaluation is no longer appropriate.

---

## TODO

1. ~~**Fix BUG 2**~~ — ✅ Fixed (2026-06-19)

2. ~~**Gradient-flow fix (Stages 0–2)**~~ — ✅ Done (2026-06-26). gen_grad_norm
   instrumentation, soft embeddings, GP rewrite, embed-the-real, KL anchor all implemented.
   Next: run seeded-mode training and validate dynamics (gen_grad_norm > 0, kl_loss stable).

3. ~~**Fix remaining bugs (3–6)**~~ — ✅ Fixed (2026-06-19)

4. ~~**Add the uniqueness metric (QoL 1)**~~ — ✅ Fixed (2026-06-19). QoL 2–3 also done.

5. **Verify blind mode diversity** — run a small generation test (e.g. 1k sequences) and
   confirm unique sequence count is well above ~1 per length now that multinomial is in place.

6. **Re-run the 30 AF3 error sequences** — these are unevaluated potential candidates.

7. **Re-run 120k eval with relaxed filter** — drop `scaccuracy` threshold or remove entirely.
   Recover the second-best candidate (5.425 Å RMSD) that was incorrectly filtered out.

8. **Retrain 2-3 epochs on best hyperparams** — once the gradient path is fixed, assess whether
   training dynamics change meaningfully before committing to a full large-scale retrain.

9. **Full new large-scale generation and evaluation** — after confirming the fixes work.

10. **Explore uniform top-k sampling** — suggested by Gökay (lab member). Sample uniformly from
    the top-k most probable tokens instead of proportionally from the full distribution. Similar
    to EvoDiff. Avoids near-zero probability tokens while keeping diversity. Optionally make k
    adaptive based on critic feedback (widen when critic says fake, narrow when it says real).
    Worth evaluating against `torch.multinomial` after the gradient path is fixed.

11. **Critic hard/soft embedding mismatch** — critic only ever trains on hard embeddings but
    scores a 50% hard / 50% soft blend during generator updates; possible confidence-inflation
    reward-hack, not yet observed but not ruled out. Full writeup + proposed diagnostic:
    `docs/GENERATOR_GRADIENT_FIX.md` → "Open risks / caveats".

12. ~~**KL identity test**~~ — ✅ Done (2026-07-22). `tests/check_kl_identity.py` confirms
    `compute_kl_anchor` plumbing is correct (KL(P‖P)=0 + negative control).

13. ~~**`mask_token_id` consistency**~~ — ✅ Done (2026-07-22, commit `718e446`). Confirmed `=4`
    everywhere it's used; `generate.py`'s hardcoded-default gap fixed.

14. ~~**Pin temperature per run**~~ — ✅ Done (2026-07-24). Was randomized every step
    (`models.py:131` TODO, now removed), adding noise to gen_grad_norm/kl_loss when comparing
    across λ_kl values. Replaced with a fixed `--temperature` CLI flag (default `1.0`) in
    both training scripts — no randomized-range option kept; existing diversity comes from
    `torch.multinomial` sampling and per-step remask-position randomness, not from varying
    temperature. Prerequisite for the lambda_kl sweep below.

18. **lambda_kl sweep (Phase 1 done, Phase 2 pending, 2026-07-25)** — two-phase sweep on MN5,
    seeded mode only, n_critic=4 fixed. Phase 1 (5 epochs × `lambda_kl ∈ {0, 0.005, 0.01, 0.05,
    0.1}`) complete and analyzed: `lambda_kl=0` deprioritized (critic saturated at
    `critic_loss=5.000`, `gen_grad_norm` collapsed to ~1e-7 for epochs 3-5 — quality metrics
    declined despite `unique_ratio=1.0`, consistent with the earlier Stage 3 finding). Top-2
    for Phase 2: `{0.005, 0.05}` — `0.005` the clear front-runner (only arm stable-to-improving
    on all 4 quality metrics), `0.05` a thin-margin second over `0.1`. Phase 2 script
    (`slurms/sweeps/lambda_kl_sweep_p2.sh`) filled in, not yet submitted. Design and full
    results: `docs/sweeps/lambda-kl-sweep-2026-07.md`, following `docs/RUN_DOCUMENTATION.md`'s
    convention.

15. **Reconsider the fixed 50% remask fraction** — `models.py:136` TODO. The λ=0.01 run's
    gentle decline (progres 0.92→0.87, pairwise_tm 0.75→0.64 over 3 epochs) raises whether
    that's an early-training transient or a structural rate issue tied to the fixed fraction.

16. **Validate blind mode under the gradient fix** — Stage 3 validation (item 2) covered
    seeded mode only. Blind mode is documented as higher mode-collapse risk and was never
    stable pre-fix — real gap, not a nice-to-have.

17. **Reconcile straight-through decision vs. implementation** — the Decision table
    (`docs/GENERATOR_GRADIENT_FIX.md` line 56) says straight-through was adopted for
    intermediate refinement-step commits; `compute_soft_embeds` actually does K=1 truncation
    with no straight-through. Confirm this is a deliberate simplification, not drift.

---

## Research Context

- **Lab:** Hacettepe University BioDataSciLab, Prof. Tunca Doğan
- **Conferences:** ISMB/ECCB 2025 (Liverpool, poster), HIBIT 2025 (Istanbul, poster)
- **Collaborators:** Karaca Lab (İzmir) — provided DNA-SAM distance metric and 7 Å threshold
- **DNMT dataset:** 52,637 curated sequences from UniProtKB/Swiss-Prot, IPR001525 domain
- **Target metric for functional interaction:** at least one DNA-SAM distance <= 7.25 Å
  (relaxed from the original 7.0 Å threshold used by Karaca Lab)
