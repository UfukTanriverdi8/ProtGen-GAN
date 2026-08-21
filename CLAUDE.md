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

## ✅ Resolved Issues (historical — full narrative in `docs/HISTORY.md` Phase 5/6)

All fixed and merged. Kept here as a short index of what to search for; the "why" and
full before/after detail lives in `docs/HISTORY.md`, and `docs/GENERATOR_GRADIENT_FIX.md`
/ `docs/GRADIENT_FIX_EXPLAINED.md` own the gradient-flow implementation depth.

- **Temperature/argmax bug** (`models.py` → `Generator.generate()`) — `.max()` after
  softmax was pure argmax; temperature was inert for the entire project's history until
  fixed. Fix: `torch.multinomial()`.
- **Generator gradient flow** (`models.py`/`loss.py`, all 3 stages, `docs/GENERATOR_GRADIENT_FIX.md`) —
  discrete token IDs blocked all adversarial gradient into the generator; every run before
  this fix used an effectively static generator. Fix: soft embedding pass-through
  (`compute_soft_embeds()`, 50% remask) + KL anchor against a frozen reference ProtBERT
  (`compute_kl_anchor()`). Validated via `gen_grad_norm > 0` and bounded `kl_loss` (0–4)
  in reference runs `xwres3cz`/`ydsjq9f4`. `tests/check_kl_identity.py` independently
  verifies `compute_kl_anchor`'s plumbing — run it before trusting future changes there.
- **`compute_gradient_penalty`** (`loss.py`) — bypassed the full transformer and ignored
  padding masks; now interpolates in embedding space through the real critic function
  (accepts pre-computed `real_embeds`/`fake_embeds`, `[B,L,H]`).
- **`generate.py` `mask_token_id`** — relied on `Generator`'s hardcoded class default
  instead of the tokenizer's actual mask token id (`718e446`).
- **Bugs 2–6** (2026-06-19 audit) — pLDDT tuple logged as scalar instead of unpacked
  (`10p_train.py`/`fully_masked_train.py`), dead temperature variable in
  `generate_fake_sequences` (`val_metrics.py`), wrong attention mask (`mask_token_id`
  instead of `pad_token_id`) in blind mode, an NaN-safe average immediately overwritten
  by an unsafe one, and `generate_fake_batch` defaulting to `debug=True`. All fixed same day.

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

> **⚠️ No seed was pinned until 2026-07-31** (`--seed` flag) — every comparison above, plus every
> hyperparameter sweep before that date (including lambda_kl Phases 1/2), ran with unseeded
> randomness as an unquantified confound. Treat pre-2026-07-31 "best config" conclusions as
> suggestive, not confirmed. See TODO 20 and `docs/sweeps/lambda-kl-sweep-2026-07.md`.

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

**Scoped `chore` prefixes** for non-code artifacts that don't warrant `feat`/`fix`:
- `chore(slurm)` — new or edited SLURM job scripts (`slurms/**`)
- `chore(run-logs)` — `.out`/`.err` logs pulled from MN5 into `outputs/`

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

> **Item numbers are permanent once assigned** — never renumbered, never reused, even after
> full resolution. Other docs (`docs/runs/`, `docs/investigations/`, `docs/sweeps/`), SLURM
> script comments, `wandb_tags`, and git commit messages cross-reference items by number; a
> renumber would silently break all of them. Mark an item resolved in place (✅ prefix) or fold
> its prose into `docs/HISTORY.md`/a "Resolved Issues" pointer — don't remove or renumber it.

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

14. ~~**Pin temperature per run**~~ — ✅ Done (2026-07-24), **gap found and closed 2026-07-30**.
    Was randomized every step (`models.py:131` TODO, now removed), adding noise to
    gen_grad_norm/kl_loss when comparing across λ_kl values. Replaced with a fixed
    `--temperature` CLI flag (default `1.0`) in both training scripts — no randomized-range
    option kept; existing diversity comes from `torch.multinomial` sampling and per-step
    remask-position randomness, not from varying temperature. Prerequisite for the lambda_kl
    sweep below.
    **Gap:** the 2026-07-24 fix only pinned temperature in the training-loop generation path
    (`compute_soft_embeds`/`generate_fakes_for_batch`/`generate_fake_batch`). It missed
    `generate_fake_sequences` (`val_metrics.py`), used by `run_evaluation()` in both training
    scripts to generate the sequences behind `plddt`/`scAccuracy`/`progres`/`pairwise_tm`/
    `unique_ratio` — this function still drew a fresh `uniform(0.8, 1.2)` temperature every
    fill step and had no `temperature` parameter to override it. This means every quality
    metric from the entire lambda_kl sweep (both phases) was measured under uncontrolled
    random temperature, adding an unquantified confound on top of the sampling noise already
    flagged for close calls. `gen_grad_norm`/`kl_loss`/`critic_loss`/`generator_loss` were
    NOT affected (correctly used the pinned value throughout) — the mechanistic case for
    `lambda_kl=0.05` (item 18) stands. Fixed 2026-07-30: `generate_fake_sequences` now takes
    a `temperature` parameter, both training scripts pass `args.temperature` through.

18. ~~**lambda_kl sweep**~~ — ✅ Done, reproduced twice (Phase 1/2 on 2026-07-26, retried
    2026-07-31 after fixing an unrelated seed-pinning bug — see item 20). **Winner:
    `lambda_kl=0.05`** — beat `0.005` on 3/4 quality metrics and kept `gen_grad_norm` healthy
    after critic saturation in both the original and retry runs, while `0.005` showed
    contradictory dynamics between its two runs (persistent saturation + weak gradient
    originally, no persistent saturation + wild gradient spikes on retry). Usable checkpoint:
    `/gpfs/projects/etur29/ufuk/gan-checkpoints/kl-sweep-p2-retry-kl0.05/epoch_15/`. Full
    per-phase design, all metric numbers, and the MN5 disk-quota checkpoint-save failure that
    hit the original Phase 2 run: `docs/sweeps/lambda-kl-sweep-2026-07.md`.

19. **Checkpoint storage bloat** — `10p_train.py`/`fully_masked_train.py` save a full checkpoint
    (both ProtBERT copies + both optimizer states, ~8.2-8.5 GB) every epoch, to a new `epoch_N/`
    directory, with no cleanup — this caused item 18's Phase 2 checkpoint-save failure by
    exhausting MN5's `gpfs_projects` quota (4.20 TB). The optimizer-state half of that cost
    (~2/3 of the total) is currently pure waste: it was added for a future resume script
    (QoL 5, above) that was never built. Two independent fixes worth doing before the next
    multi-epoch sweep or long run: (1) stop saving optimizer state until the resume script
    exists, (2) only keep the last checkpoint (or last N) instead of every epoch. Not yet
    implemented — discussed 2026-07-26, deferred.

20. ✅ **Critic saturation investigated (2026-08-21)** — across nearly every `lambda_kl` sweep
    arm (item 18), `critic_loss` pins at exactly `5.0000` for the remainder of training.
    **Answer: genuine degenerate collapse, confirmed via held-out validation tooling (item 22),
    and universal/architectural rather than `lambda_kl`-dependent.** `critic_loss ≈ lambda_gp`
    is the loss formula's exact arithmetic signature of a critic that has stopped being a
    function of its input (Wasserstein term → 0, gradient penalty → 1); `*_score_std` collapses
    to ~1e-5–1e-6 on both training and held-out real sequences in lockstep with the pinning,
    ruling out both benign convergence and training-set memorization. Re-running both Phase 2
    finalists (`lambda_kl=0.005`, `0.05`) side by side showed collapse occurs in both — but
    `lambda_kl` clearly affects *post-collapse stability*: `0.05` stays flat/collapsed the whole
    run, while `0.005` has a real, still not fully explained instability episode around epoch 12
    (`gen_grad_norm` spikes to 116, `generator_loss` sign-flips and never recovers by run end).
    This means the original "0.05 wins" sweep conclusion (item 18) was partly based on
    critic-derived signal that was itself degenerate during measurement — the more defensible
    basis is structural quality metrics plus this diagnostic's downstream-stability finding, not
    the original `gen_grad_norm`/critic-loss dynamics. Full analysis, anomaly timeline, and the
    unconfirmed AdamW-overshoot hypothesis for the epoch-12 event:
    `docs/investigations/critic-saturation-diagnostic-2026-08.md`. Discovery narrative for the
    underlying saturation/seed issues: `docs/HISTORY.md` Phase 6.
    **Not yet done:** test whether `lambda_gp=5` is too strong (cheap "ignore everything"
    optimum); directly test the AdamW-overshoot hypothesis; check whether other `lambda_kl`
    values reproduce the same universal-collapse/`lambda_kl`-dependent-stability pattern.

21. **wandb `config` field empty for offline runs — root cause found (2026-07-31)** — every run
    in the lambda_kl sweep shows `run.config` (via both MCP and the direct Python API)
    containing only `_wandb` client metadata, no real hyperparameters, despite both training
    scripts correctly calling `wandb.config.update({...})` with the right values right after
    `wandb.init()`. Inspecting the raw `config.yaml` written locally by an offline run *before
    any sync ever touches it* (`arrivals/offline-run-20260730_132036-w8c2wl85/files/config.yaml`
    and its `-2rohdbx2` sibling) confirms neither has any top-level keys besides
    `wandb_version`/`_wandb` — so **offline mode never materializes `wandb.config.update()`'s
    values into `config.yaml` at all**, not an MCP or sync-step artifact as two earlier notes
    wrongly concluded (both mistook wandb's auto-captured CLI-args telemetry, buried at
    `_wandb.e.<hash>.args`, for the real config field). Worth testing whether passing
    hyperparameters via `wandb.init(config={...})` directly (instead of a separate `.update()`
    call after init) persists correctly under `mode="offline"` — try this on the next MN5 run
    regardless of sweep status. Every hyperparameter value already recorded in this project's
    run docs is still correct (backfilled from the submitting SLURM script), just not
    confirmable via wandb's own config field until this is fixed. See
    `docs/sweeps/lambda-kl-sweep-2026-07.md`'s Results section for the full trail.

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

22. **No validation loss exists — only training loss** (raised 2026-07-31). Neither training
    script holds out any data: `dataset.py`'s `get_dynamic_dataloaders` splits each epoch's data
    by *role* (gen vs critic), re-shuffled fresh every epoch — not a train/val partition. Every
    `critic_loss`/`generator_loss` logged to wandb (baseline/mid/end tags, see `run_evaluation()`
    in both scripts) is a training-batch value. The structural metrics (`plddt`/`scAccuracy`/
    `progres`/`pairwise_tm`/`unique_ratio`) are a legitimate held-out signal for the *generator*
    (computed on freshly generated sequences, never touching training data) but say nothing about
    whether the *critic* generalizes — it could be scoring real-vs-fake well/badly on training
    reals purely from memorization, not genuine discrimination. At minimum, think through whether
    holding out a slice of real sequences (never shown to the critic during training) and
    periodically scoring them would give a meaningful signal here — directly useful for item 20's
    saturation question too (does saturation persist on real sequences the critic never trained
    on?). Note: WGAN-GP's `critic_loss` isn't a bounded/normalized loss like a classification
    loss, so a held-out version of it won't have the classic "val loss diverges from train loss"
    overfitting signature — same saturation question, different-looking curve. Not yet
    implemented or even fully designed — this item is "consider the possibility," not a decided
    approach.
    **Implemented for the critic, `10p_train.py` only (2026-08-20):** `--holdout_size`
    (default 200) reserves that many sequences from `dnmt_full.txt`'s pool at load time,
    before `get_dynamic_dataloaders`'s per-epoch reshuffle ever sees them — carved in-script
    via a dedicated seeded `torch.Generator` rather than a persisted file, so it stays
    reproducible across runs with the same `--seed` without adding a second, independently-
    drifting data-split artifact (the codebase already has one such artifact:
    `file_formatter.py`'s dead train/val split is independent from its `dnmt_gen`/`dnmt_critic`
    split, so a revived `dnmt_val.txt` could silently overlap `dnmt_critic.txt`). Note: because
    the held-out indices are `perm[:holdout_size]` from a seeded permutation, two runs with the
    same `--seed` and same effective pool size (i.e. same `--max_train_seqs`) get *nested*
    holdout sets when `--holdout_size` differs — the smaller holdout's sequences are a prefix
    subset of the larger's, not a disjoint set — but the sets still aren't identical, so
    `holdout_*` metrics aren't directly comparable across arms with different `--holdout_size`;
    hold it constant across a sweep if comparing those metrics between arms.
    Shrinks `10p_train.py`'s effective training pool by `holdout_size` relative to every prior run —
    a <1% change at the default value against ~52k sequences, called out in the flag's `help=`
    text. `--holdout_eval_fakes` optionally also scores fresh generator output against the
    held-out set for real/fake symmetry. See item 20 for what's logged and the diagnostic
    rationale. **Not done:** `fully_masked_train.py` (blind mode) has no equivalent — deferred
    since blind mode isn't validated yet (item 16). The unified-script merge (QoL 5) was also
    explicitly deferred rather than bundled with this change.

23. **Re-evaluate the evaluation pipeline itself** (from 24 Jul 2026 notes, not previously
    tracked). The current metric set (`pLDDT`/`scAccuracy`/`progres`/`pairwise TM`/
    `seq_similarity`) was chosen a long time ago. Consider alternatives — **ProTrek** in
    particular, plus ESMFold2 — to see whether they'd give a more reliable quality/functional
    signal than the current set, especially given `progres`/`scAccuracy` are already known to be
    poor predictors of DNA-SAM binding (see Key Evaluation Findings above). Not yet scoped —
    needs a look at what ProTrek actually outputs and whether it's a drop-in replacement or
    additive.

---

## Research Context

- **Lab:** Hacettepe University BioDataSciLab, Prof. Tunca Doğan
- **Conferences:** ISMB/ECCB 2025 (Liverpool, poster), HIBIT 2025 (Istanbul, poster)
- **Collaborators:** Karaca Lab (İzmir) — provided DNA-SAM distance metric and 7 Å threshold
- **DNMT dataset:** 52,637 curated sequences from UniProtKB/Swiss-Prot, IPR001525 domain
- **Target metric for functional interaction:** at least one DNA-SAM distance <= 7.25 Å
  (relaxed from the original 7.0 Å threshold used by Karaca Lab)
