# ProtGen — GAN Repo Context for Claude Code

## Commands

### Environment Setup
```bash
# Required on every machine before running any script.
# MN5:  add to your SLURM job script
# Anzu: add to ~/.bashrc
export SOURCE_DIR=/gpfs/projects/etur29/ufuk   # MN5 example
export SOURCE_DIR=/path/to/your/models          # Anzu example
# config.py resolves PROTBERT, ESMFold, and checkpoint paths from SOURCE_DIR.
# MN5 hostnames are auto-detected as a fallback if SOURCE_DIR is not set.
```

### Training (MN5 — submit via SLURM)
```bash
sbatch long_10p_run.sh        # Seeded mode extended run
sbatch lr_full_run.sh         # Blind mode LR grid search

# Direct invocation (Anzu / local debugging)
python 10p_train.py --run_name debug_run --n_epochs 2 --batch_size 4
python fully_masked_train.py --run_name debug_run --n_epochs 2 --batch_size 4
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
sbatch mass_eval.sh                              # Batch structural eval (120k+ seqs)
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

## ⚠️ ARCHITECTURAL ISSUE — Generator Has Never Received Adversarial Gradient

**Affects:** Every training run ever. Not a one-line fix — requires architectural change.

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

### Fixes (all require architectural change)

- **Gumbel-Softmax relaxation:** Instead of sampling discrete tokens, use
  `F.gumbel_softmax(logits / temperature, tau=1.0, hard=False)` to get soft
  pseudo-one-hot vectors. Pass these as weighted sums of the embedding matrix to the critic:
  `soft_embeds = soft_tokens @ critic.protbert.bert.embeddings.word_embeddings.weight`
  Gradients flow through the soft approximation back into the generator.

- **Soft embedding pass-through (simpler):** After the generator produces logits, compute:
  `soft_embeds = F.softmax(logits / temperature, dim=-1) @ embedding_matrix`
  Skip the discrete sampling for the critic pass; keep sampling only for the actual
  sequence output. Gradients flow through `softmax → embedding matrix → generator logits`.

- **REINFORCE / policy gradient:** Treat the generator as a policy, use the critic score
  as reward signal, apply REINFORCE gradient estimate. No differentiable path needed, but
  higher variance — requires careful baseline tuning.

The Gumbel-Softmax or soft-embedding approach is the most natural fit given the existing
architecture. Both would require the critic to accept continuous 3D embeddings (which it
already supports via the dim==3 branch in `models.py`).

---

## ✅ FIXED — `compute_gradient_penalty` call signature (`loss.py` + both training scripts)

`loss.py` was updated to add `real_mask` and `fake_mask` parameters (commit 8ebec32), and
both training scripts have been updated to match:

```python
gp = compute_gradient_penalty(
    critic, real_data, fake_data,
    attn_mask_real,
    (fake_data != tokenizer.pad_token_id).long(),
    device
)
```

---

## ⚠️ BUGS FOUND — Not Yet Fixed (as of 2026-04-24)

These were identified by auditing the codebase after the temperature fix. Fix all of these
before running any new training.

---

### BUG 2 — `calculate_plddt_scores_and_save_pdb` returns a tuple; training scripts treat it as a scalar

**Files:** `val_metrics.py:207`, `10p_train.py:231`, `fully_masked_train.py:162`
**Severity:** CRITICAL — silently logs a tuple to wandb instead of a float; all pLDDT
metrics in every training run are garbage

`val_metrics.py` returns `(avg_plddt_score, plddt_scores)`. Both training scripts do:

```python
avg_plddt_score = calculate_plddt_scores_and_save_pdb(...)
wandb.log({"plddt_score": avg_plddt_score, ...})
```

`avg_plddt_score` is a `(float, list)` tuple — wandb receives a tuple, not a number.

Fix: unpack the return value in both training scripts:
```python
avg_plddt_score, _ = calculate_plddt_scores_and_save_pdb(...)
```

---

### BUG 3 — `generate_fake_sequences` in `val_metrics.py` ignores its own random temperature

**File:** `val_metrics.py:119–132`
**Severity:** Significant — evaluation sequences during training always use temperature=1.0;
the random temperature variation is computed but silently discarded

```python
fixed_temp = 1.0
while current_masking_rate > 0:
    random_temp = torch.empty(1).uniform_(0.8, 1.2).item()  # computed but never used
    generated_ids = generator.generate(..., temperature=fixed_temp)  # always 1.0
```

Fix: replace `fixed_temp` with `random_temp` in the `generator.generate()` call.

---

### BUG 4 — `fully_masked_train.py` attention mask during generation excludes [MASK] tokens

**File:** `fully_masked_train.py:224`
**Severity:** Significant — mask tokens are invisible to the rest of the sequence during
attention, degrading generation quality

```python
# BROKEN:
updated_attention_mask = (final_input_ids != tokenizer.mask_token_id).long()

# CORRECT (as in 10p_train.py):
updated_attention_mask = (final_input_ids != tokenizer.pad_token_id).long()
```

Zeroing out [MASK] positions in the key/value attention mask means no token can attend to
those positions as context. ProtBERT is designed to predict masked tokens — they must be
visible as keys in bidirectional attention.

---

### BUG 5 — NaN guard in `calculate_plddt_scores_and_save_pdb` is silently overwritten

**File:** `val_metrics.py:200–206`
**Severity:** Moderate — if any ESMFold pLDDT is NaN, the average logged to wandb will be NaN

```python
valid = [x for x in plddt_scores if isinstance(x, (float, int)) and not np.isnan(x)]
avg_plddt_score = (sum(valid) / len(valid)) if valid else -1  # NaN-safe
# Then immediately overwritten:
if len(plddt_scores) > 0:
    avg_plddt_score = sum(plddt_scores) / len(plddt_scores)  # NaN-unsafe
```

Fix: delete the second `if` block; keep only the NaN-filtered `valid` computation.

---

### BUG 6 — `generate_fake_batch` in `fully_masked_train.py` defaults `debug=True`

**File:** `fully_masked_train.py:199`
**Severity:** Quality of life — floods SLURM logs with per-position mask counts on every
single training batch across every epoch

```python
# BROKEN:
def generate_fake_batch(..., debug=True):

# FIXED:
def generate_fake_batch(..., debug=False):
```

---

## ⚠️ QUALITY OF LIFE — Missing Features Worth Adding

### QoL 1 — Uniqueness metric missing from in-training evaluation

**Files:** `10p_train.py`, `fully_masked_train.py` → `run_evaluation()`

The eval loop tracks pLDDT, scAccuracy, progres, pairwise TM-score — none of which would
have caught mode collapse early. A simple uniqueness ratio on the eval batch would have
caught the argmax bug on the very first run.

Add to `run_evaluation()`:
```python
unique_ratio = len(set(generated_sequences)) / len(generated_sequences)
wandb.log({"unique_ratio": unique_ratio, ...})
```

This is a 1-line addition and is the single most valuable early-warning signal.

### QoL 2 — Optimizer state not saved in checkpoints

**Files:** `10p_train.py:386–401`, `fully_masked_train.py:375–391`

Only ProtBERT weights and the classifier head are saved. If a run crashes and is resumed,
Adam's momentum and variance accumulators are lost, causing instability for many batches.

Add to checkpoint saving:
```python
torch.save(gen_optimizer.state_dict(),    f"{save_dir}/gen_optimizer.pth")
torch.save(critic_optimizer.state_dict(), f"{save_dir}/critic_optimizer.pth")
```

And add corresponding load logic at the start of any resume script.

### QoL 3 — `sample_sequence_length` re-reads the full dataset file on every call

**File:** `val_metrics.py:48–57`

`sample_sequence_length()` opens, reads, and parses the entire dataset file from disk
every single invocation. It is called once per generated sequence during evaluation
(10+ calls per `run_evaluation()`). Fix: load the length list once at module level or
pass it as an argument.

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
├── generate_samples.sh        Array job: generate 10k sequences from 6 checkpoints
│                              (full + seeded in parallel per checkpoint)
├── mass_eval.sh               Array job: parallel metric evaluation of generated seqs
├── long_10p_run.sh            Extended training run for seeded mode
├── lr_10p_run.sh              LR grid search for seeded mode (25 combinations)
├── lr_full_run.sh             LR grid search for blind mode
├── lr_best_runs.sh            Retraining with best LRs from sweep
├── n_critic_10p_run.sh        n_critic grid search (seeded mode)
└── n_critic_full_run.sh       n_critic grid search (blind mode)

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
├── Conda-Environment-for-ProtGEN_mn5.yml  Full conda env spec (PyTorch 2.4.1,
│                              transformers 4.46, ESMFold, ProteinMPNN, PROGRES, etc.)
└── bfg-1.15.0.jar             BFG repo cleaner (git history cleanup utility)
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

---

## Infrastructure

### Key Documents
- `docs/HISTORY.md` — full project narrative: every phase, architectural decision, bug discovery, and current state. Read before suggesting experiments or evaluating what's been tried.
- `docs/GIT_WORKFLOW.md` — complete two-remote git workflow and wandb offline sync. Includes agent-specific notes at the bottom.

### Claude Code Automation (`.claude/`)
- **Hook** — blocks edits to `.env` and `Conda-Environment-for-ProtGEN_mn5.yml`
- **Skill: `slurm-job`** — generates MN5 SLURM scripts from run parameters
- **Skill: `bug-fix-checklist`** — Claude-only; greps for all known unfixed bugs before touching training/eval files

| Environment | Purpose |
|-------------|---------|
| **Anzu** | Hacettepe BioDataSciLab GPU server (Ubuntu). Used for eval runs, debugging, smaller experiments. SSH access. |
| **MareNostrum5 (MN5)** | BSC supercomputer, thousands of H100s. Used for large GAN training runs and AF3 evaluation. **No internet access** — files transferred via SCP (upload) and SFTP (download). |

### Git Workflow (Two-Remote Setup)
MN5 has no internet access, so it cannot push/pull directly to GitHub. The local machine
(or Anzu) acts as a mediator:

- `origin` → GitHub (remote)
- `mn5` → MN5 local repo (remote)

Typical flow: develop locally → push to `origin` (GitHub) and/or push to `mn5` directly.
To sync MN5 with GitHub, pull from `origin` locally then push to `mn5`, or vice versa.

### wandb Workflow on MN5
MN5 has no internet. wandb runs are logged offline, then:
1. Download wandb run directory via SFTP to Anzu
2. Upload to wandb from Anzu with `wandb sync`

### Evaluation Frequency
The evaluation frequency inside the training loop should be dynamic and proportional to
`n_critic`, since dataset distribution is split dynamically per epoch. Static every-250-batch
evaluation is no longer appropriate.

---

## Possible Next Steps

1. **Fix BUG 2** — training logs a tuple to wandb instead of a float; all pLDDT metrics are garbage.
   Unpack the return value in both training scripts before any new run.

2. **Implement differentiable generator-to-critic path** — the GAN has never functioned as a GAN
   because discrete token IDs break the gradient path (see Architectural Issue section).
   Recommended approach: soft embedding pass-through using `softmax(logits) @ embedding_matrix`
   for the critic forward pass during generator updates.

3. **Fix remaining bugs (3–6)** — attention mask bug in blind mode, dead temperature code,
   NaN guard, debug flood. All are straightforward, see bug section.

4. **Verify blind mode diversity** — run a small generation test (e.g. 1k sequences) and
   confirm unique sequence count is well above ~1 per length now that multinomial is in place.

5. **Re-run the 30 AF3 error sequences** — these are unevaluated potential candidates.

6. **Re-run 120k eval with relaxed filter** — drop `scaccuracy` threshold or remove entirely.
   Recover the second-best candidate (5.425 Å RMSD) that was incorrectly filtered out.

7. **Retrain 2-3 epochs on best hyperparams** — once the gradient path is fixed, assess whether
   training dynamics change meaningfully before committing to a full large-scale retrain.

8. **Full new large-scale generation and evaluation** — after confirming the fixes work.

9. **Explore uniform top-k sampling** — suggested by Gökay (lab member). Sample uniformly from
   the top-k most probable tokens instead of proportionally from the full distribution. Similar
   to EvoDiff. Avoids near-zero probability tokens while keeping diversity. Optionally make k
   adaptive based on critic feedback (widen when critic says fake, narrow when it says real).
   Worth evaluating against `torch.multinomial` after the gradient path is fixed.

---

## Research Context

- **Lab:** Hacettepe University BioDataSciLab, Prof. Tunca Doğan
- **Conferences:** ISMB/ECCB 2025 (Liverpool, poster), HIBIT 2025 (Istanbul, poster)
- **Collaborators:** Karaca Lab (İzmir) — provided DNA-SAM distance metric and 7 Å threshold
- **DNMT dataset:** 52,637 curated sequences from UniProtKB/Swiss-Prot, IPR001525 domain
- **Target metric for functional interaction:** at least one DNA-SAM distance <= 7.25 Å
  (relaxed from the original 7.0 Å threshold used by Karaca Lab)