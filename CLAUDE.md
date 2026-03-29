# ProtGen — GAN Repo Context for Claude Code

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

- Uses WGAN-GP (gradient penalty λ=10) — the GP implementation is a known trap worth verifying in `full_train.py` and `10p_train.py`
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

### DrugGEN Project Repo File Structure

Just for reference here is the file structure of the DrugGEN repo:

```text
├── assets/
│   ├── (placeholder for future assets: diagrams, figures, etc.)
├── data/
│   ├── decoders/
│   │   └── .gitkeep
│   ├── encoders/
│   │   └── .gitkeep
│   └── .gitkeep
├── experiments/
│   ├── inference/
│   │   └── .gitignore
│   ├── logs/
│   │   └── .gitignore
│   ├── models/
│   │   ├── DrugGEN-akt1/
│   │   │   └── .gitkeep
│   │   ├── DrugGEN-cdk2/
│   │   │   └── .gitkeep
│   │   └── NoTarget/
│   │       └── .gitkeep
│   ├── results/
│   │   ├── .gitignore
│   │   └── tensorboard.txt
│   └── samples/
│       └── .gitignore
├── results/
│   ├── docking/
│   │   ├── (docking results files stored as CSVs here)
│   ├── generated_molecules/
│   │   ├── DrugGEN_generated_molecules_AKT1.csv
│   │   ├── DrugGEN_generated_molecules_CDK2.csv
│   │   └── Selected_denovo_AKT1_inhibitors.csv
│   ├── evaluate.py
│   └── README.md
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── utils.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── layers.py
│   │   ├── loss.py
│   │   └── models.py
│   ├── util/
│   │   ├── __init__.py
│   │   ├── smiles_cor.py
│   │   └── utils.py
│   └── __init__.py
├── .gitignore
├── environment.yml
├── inference.py
├── LICENSE
├── README.md
├── setup.sh
└── train.py
```

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

### Blind Mode (`full_train.py`)
- Critic trains on the entire dataset.
- Generator inputs: fully masked sequences of lengths sampled from the dataset distribution.
- No seed — the generator must produce a sequence from scratch.
- File naming convention: runs include `full` in the name.

---

## ⚠️ CRITICAL BUG — Fix Before Any New Runs

**File:** `models.py`
**Method:** `Generator.generate()`

### The Bug

```python
# CURRENT (BROKEN):
logits = outputs.logits / temperature
probabilities = F.softmax(logits, dim=-1)
confidence, predicted_ids = probabilities.max(dim=-1)  # ← BUG: pure argmax
```

Temperature is applied to the logits before softmax, which correctly reshapes the probability
distribution. But then `.max()` (argmax) is called — which always picks the single highest
probability token regardless of the distribution shape. **Dividing logits by any positive
constant never changes which index is the maximum.** Temperature has literally zero effect
on which token gets selected.

### Consequences

- **Blind mode:** every sequence starts as 100% masks, same model, deterministic argmax →
  identical output for every sequence of the same length. In practice: ~300 unique sequences
  out of 10,000 generated, one per sampled length. Total mode collapse.
- **Seeded mode:** different seeds produce different inputs → different logits → different
  argmax. Seeded mode was *accidentally* functional due to seed variation, not because
  sampling was working.
- **GAN training:** fake sequences shown to the critic during training were always greedy/
  deterministic. The critic never saw diverse fakes. The adversarial dynamic was crippled
  from the start. This is why training metrics never meaningfully improved across ~60-70 runs.

### The Fix

```python
# FIXED:
logits = outputs.logits / temperature
probabilities = F.softmax(logits, dim=-1)

# Sample from the distribution instead of taking argmax:
predicted_ids = torch.multinomial(
    probabilities.view(-1, probabilities.size(-1)),
    num_samples=1
).view(batch_size, seq_len)
confidence = probabilities.gather(-1, predicted_ids.unsqueeze(-1)).squeeze(-1)
```

After this fix:
- Temperature actually controls diversity: lower = more conservative, higher = more exploratory.
- Blind mode produces genuinely diverse sequences.
- GAN training sees varied fakes → the adversarial loop can actually function.

**This is a one-line conceptual fix but affects both training and inference simultaneously.**
Verify blind mode diversity after applying it with a small generation test before any large run.

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
│                              ⚠️  Contains the temperature/argmax bug — see fix section above
├── loss.py                    Wasserstein loss + gradient penalty for WGAN training
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

## Possible Next Steps (post-bug-fix)

1. **Apply `torch.multinomial()` fix in `models.py`** — one-line change, highest priority.
2. **Verify blind mode diversity** — run a small generation test (e.g. 1k sequences) and
   confirm unique sequence count is no longer ~1 per length.
3. **Re-run the 30 AF3 error sequences** — these are unevaluated potential candidates.
4. **Re-run 120k eval with relaxed filter** — drop `scaccuracy` threshold to ~0.25 or remove
   entirely. Recover the second-best candidate that was incorrectly filtered out.
5. **Retrain 2-3 epochs on best hyperparams** — assess whether the fix changes training
   dynamics before committing to a full large-scale retrain.
6. **Full new large-scale generation and evaluation** — after confirming the fix works.
7. **Explore uniform top-k sampling as an alternative to `torch.multinomial`** — suggested by
   Gökay (lab member). Instead of sampling proportionally from the full softmax distribution,
   sample uniformly from the top-k most probable tokens at each position. Similar to the
   approach used in EvoFiff. This gives exploration while staying within plausible amino acids
   and avoids ever picking near-zero probability tokens. A further idea is to make k adaptive
   based on critic feedback — widen k when the critic classifies the sequence as fake, narrow
   it when the sequence looks real. Worth evaluating against `torch.multinomial` after the
   basic fix is confirmed working.

---

## Research Context

- **Lab:** Hacettepe University BioDataSciLab, Prof. Tunca Doğan
- **Conferences:** ISMB/ECCB 2025 (Liverpool, poster), HIBIT 2025 (Istanbul, poster)
- **Collaborators:** Karaca Lab (İzmir) — provided DNA-SAM distance metric and 7 Å threshold
- **DNMT dataset:** 52,637 curated sequences from UniProtKB/Swiss-Prot, IPR001525 domain
- **Target metric for functional interaction:** at least one DNA-SAM distance <= 7.25 Å
  (relaxed from the original 7.0 Å threshold used by Karaca Lab)