# protgen-gan - Entire History

**Reconstructed from the meeting logs, code commits, and documentation of the project.**  
**Last Updated: 14 May 2026**

This document captures the full life of the project: every major architectural decision, what was learned from each phase, and every significant bug discovered. It is written as a research narrative rather than a changelog, because the most important things to preserve are not just what changed but why, and what the consequences were.

The project does not yet have a formal published name. "protgen-gan" is the working codename.

---

## Background and Motivation

This project is developed at Hacettepe University BioDataSciLab under Prof. Tunca Doğan, started in the summer of 2024 as a TUBITAK STAR internship and continued as voluntary research. The goal is to generate functional artificial protein sequences that mimic the DNMT family(DNA methyltransferases), enzymes that catalyse cytosine methylation and whose misregulation is linked to cancer and developmental disorders.

The core idea was to combine a fine-tuned protein language model with a GAN-like adversarial architecture: rather than designing proteins by hand or relying on pure sampling from a pretrained model, train a critic to push the generator toward producing sequences that look like real DNMT proteins.

The project was presented as a poster at ISMB/ECCB 2025 in Liverpool and at HIBIT 2025 in Istanbul. A TUBITAK 2224-B grant supported conference attendance. The Karaca Lab in İzmir served as an external collaborator, providing the DNA-SAM distance metric and the 7 Å threshold used as the final functional filter.

---

## Phase 1: ProtBERT Fine-tuning (September - October 2024)

The backbone chosen for the whole project was ProtBERT, a masked protein language model based on the BERT architecture pretrained on general protein datasets. The decision was to fine-tune it in two successive stages before using it in the GAN.

**Stage 1 -- Broad enzyme pre-training.** ProtBERT was fine-tuned on a large-scale enzyme dataset of approximately 600,000 sequences derived from EC 2.1.1 transferases (methyltransferase-like enzymes broadly), filtered by mean ± 2×std over sequence lengths to remove outliers. The intent was to give the model a strong methyltransferase context before narrowing to the specific target family.

**Stage 2 -- DNMT specialization.** The stage 1 checkpoint was then further fine-tuned on approximately 50,000 sequences corresponding to EC 2.1.1.37 (DNA cytosine-5-methyltransferase), sourced from UniProtKB/Swiss-Prot entries annotated with the IPR001525 domain.

**Fixed vs dynamic masking.** ProtBERT's original pretraining used a fixed 15% masking rate. During fine-tuning, a dynamic masking scheme was tested alongside it: at each training step the mask fraction was sampled from a mixture of Beta(6,6) and Beta(8,2) distributions, yielding an expected masking rate of approximately 58%. The dynamic scheme produced similar performance to fixed masking but was chosen for its better alignment with the demands of de novo generation, where the model would later need to fill in heavily masked sequences.

Both stages ran for 300 epochs each. Checkpoints were selected based on F1, accuracy, and loss: epoch 200 from the fixed-mask run and epoch 260 from the dynamic-mask run were used to seed stage 2, and after stage 2 evaluation the epoch 300 dynamic-mask checkpoint was selected as the GAN backbone.

---

## Phase 2: Initial GAN Architecture (October - December 2024)

With the fine-tuned ProtBERT checkpoint in hand, GAN development began.

**Architecture decisions.** Two copies of the fine-tuned ProtBERT were used -- one as the generator and one as the critic. The generator uses its ProtBERT instance as-is, with no modifications. The critic augments its ProtBERT backbone with a feedforward classification head (a 4-layer MLP) to score sequences as real or fake. The training framework is WGAN-GP (Wasserstein GAN with gradient penalty).

**Generation approach.** The generator does not produce a sequence in a single forward pass. Instead it works iteratively: it starts from a masked sequence, runs ProtBERT to get logit predictions for all masked positions, keeps the top 10% most confident predictions, re-masks the rest, and repeats until the sequence is fully filled. The rationale is that building on the most confident predictions first yields more coherent sequences than filling randomly.

**Two training modes were defined:**

Seeded mode (also called 10p mode) splits the dataset in half -- one half for generator training, one for the critic. Generator inputs are real sequences with 90% of their tokens masked, leaving 10% as a "seed" that guides generation. The seed provides biological context.

Blind mode (also called full mode) gives the critic the entire dataset. The generator receives a fully masked sequence of a length sampled from the dataset distribution and must produce a sequence from scratch with no seed.

**Early training instability.** The initial GAN training runs were highly unstable. Loss values were erratic, not converging to any meaningful value. Early investigations focused on gradient clipping (later removed), the loss function formulation, and ensuring the generation loop was producing varied outputs. wandb logging on MN5 was also problematic because MN5 has no internet access, requiring a manual offline-sync workflow via SFTP to Anzu.

**First-epoch freezing.** An important early architectural decision was to freeze both ProtBERT instances during the first epoch and train only the classification head. Because the ProtBERT backbones were already heavily fine-tuned for DNMT specificity, their weights were well-optimized. The classification head started from random initialization and would produce garbage feedback to the generator if used immediately. Freezing ProtBERT during epoch 1 lets the classification head warm up before the adversarial game starts.

The DrugGEN paper from the same lab (Nature Machine Intelligence 2025) provided relevant lessons during this phase. DrugGEN is a GAN-based de novo drug design system using graph transformers. Key transferable lessons: WGAN-GP is a known implementation trap worth verifying carefully; tracking uniqueness and novelty metrics throughout training rather than just loss would catch mode collapse early; transformer depth of 1 was optimal in DrugGEN and deeper backbones tend to be less stable.

---

## Phase 3: Training Stabilization and Hyperparameter Search (January - June 2025)

With a stable training loop established, the focus shifted to finding good hyperparameters. A formal sweep was not feasible for a model of this size, so combinations were tested manually.

**n_critic search.** The n_critic parameter controls how many critic update steps are taken per generator step. Values of 1, 2, 4, 8, and 12 were tested. n_critic=8 was found to be optimal and became the standard setting. With n_critic=8, the dataset was split dynamically at the start of each epoch such that 8/9 of the DNMT dataset went to the critic and the remaining 1/9 to the generator.

**Learning rate search.** With n_critic fixed at 8, learning rate combinations were tested across both modes. After extensive testing the best configurations identified were:

Seeded mode: (lr_gen=5e-6, lr_crit=5e-5) and (lr_gen=5e-5, lr_crit=5e-4).
Blind mode: (lr_gen=1e-5, lr_crit=5e-5) and (lr_gen=5e-5, lr_crit=5e-4).

**Evaluation frequency.** Evaluation had been running every 250 batches statically, but with the dynamic dataset splitting tied to n_critic, 250 batches were often not reached within an epoch. The evaluation frequency needed to be made dynamic and proportional to n_critic and the actual batch count per epoch.

**Evaluation metrics implemented.** The in-training evaluation pipeline grew over this period to include pLDDT (ESMFold structure prediction confidence), scAccuracy (self-consistency: sequence → ESMFold → ProteinMPNN → alignment score back to original), Progres (structural similarity to a reference DNMT protein), and pairwise TM-score diversity. Sequence similarity to a reference DNMT was also tracked as a filter metric.

**Training behavior observed.** Across all runs, a consistent pattern emerged. Seeded mode: loss starts at reasonable values (around 4-5), critic loss collapses to near zero within a few steps and stays there, structural metrics like pLDDT oscillate between roughly 0.65 and 0.72 without clear improvement. Blind mode: structural metrics behave similarly to seeded mode, but loss is much more unstable, frequently hitting the 200-300 range in both positive and negative directions.

At the time, this behavior was attributed to the difficulty of GAN training in general. In retrospect, these loss curves were entirely explained by the two major bugs described later.

---

## Phase 4: Large Scale Evaluation (August - November 2025)

After extensive training experiments without finding runs with significant structural metric improvement, the decision was made to move to large-scale generation and downstream evaluation with AlphaFold3.

**Checkpoint selection.** Six representative conditions were selected for large-scale generation: four of the best GAN checkpoints (two seeded mode, two blind mode, with the best LR configurations from the search) and two baselines (the fine-tuned ProtBERT with no GAN training, and the base ProtBERT with no fine-tuning).

**Generation.** For each of the six conditions, 5,000 sequences were generated in seeded mode and 5,000 in blind mode, treating the GAN as an inference pipeline. This produced 10,000 sequences per condition, 60,000 total in the first batch.

**Pre-AF3 filtering.** Before AlphaFold3 analysis, sequences were filtered by length (≤300), pLDDT (≥0.8), Progres (≥0.9), scAccuracy (≥0.3), and sequence similarity (≤0.35). This reduced 60,000 sequences to a few hundred candidates.

**AlphaFold3 on MN5.** AF3 inference was set up on MareNostrum5. The analysis used AF3 co-folding of each generated protein sequence with DNA and SAM (the cofactor involved in methylation). Rather than relying only on ipTM and pTM scores, a DNA-SAM binding distance metric was calculated -- the distance between the SAM molecule and the DNA in the folded structure. The Karaca Lab threshold: at least one distance value below 7 Å for the interaction to be considered meaningful. This was later relaxed to 7.25 Å to increase candidate count.

**First evaluation batch results.** From 60,000 sequences after filtering and AF3 analysis, a small number of survivors emerged. All seeded mode. Blind mode contributed zero meaningful sequences -- consistent with the later-discovered mode collapse from the argmax bug.

**Second evaluation batch.** A second batch of 120,000 sequences was generated from the same checkpoints. The pre-AF3 filter was relaxed -- Progres was removed as a filter criterion after it became clear it was a poor predictor of DNA-SAM binding proximity. scAccuracy was kept at ≥0.3. This produced 404 additional sequences for AF3 analysis, of which 8 returned errors.

**Key finding about Progres and scAccuracy.** The best surviving sequences (DNA-SAM distances of 5.27-6.68 Å) came from the relaxed-filter batch where both Progres and scAccuracy filters were removed. Sequences that passed strict Progres ≥ 0.9 did not outperform those that did not. This established that AlphaFold3 co-folding with binding distance is the only reliable functional filter, and Progres/scAccuracy should not be used as hard gates.

**Survivors.** Across both evaluation batches, 9 unique sequences survived the full pipeline. All 9 came from seeded mode. The best candidate was sequence 2508_len279 from the full_nc8_lrgen_5e-5_lrcrit_5e-4 checkpoint, with a DNA-SAM binding distance of 5.272 Å. This sequence was structurally validated by an external chemistry professor who confirmed DNMT-3a-like secondary structure with preserved hydrophilic interactions for SAM-mediated methylation. 30 AF3 error sequences from the 120k batch remain unevaluated and are potential candidates.

**One unexpected finding.** One of the 9 survivors came from the fine-tuned ProtBERT baseline, not from any GAN checkpoint. This was uncomfortable, as the expectation was that the GAN training would outperform the baseline. In retrospect, given the bugs described below, this outcome was inevitable.

---

## Phase 5: Bug Discovery (March 2026)

March 2026 was when the two most consequential bugs in the project were discovered, invalidating approximately 8 months of training runs.

---

### Bug 1 — Argmax / Temperature Bug (models.py)

**Discovery.** While analyzing why the 120k evaluation batch produced only 2 AF3 survivors despite 404 sequences entering the pipeline, it was found that 393 of the 731 total AF3-evaluated sequences were identical copies of the same sequence. All came from blind mode. Further investigation revealed that blind mode had only ever produced 2 distinct sequences across all its output.

Tracing back through the generation pipeline to the txt output files confirmed the duplicates were not a CSV-building artifact -- they were present in the raw generation output. The source was in `models.py`, inside `Generator.generate()`:

```python
# The broken line:
confidence, predicted_ids = probabilities.max(dim=-1)
```

Temperature was applied to logits, softmax was run to produce a probability distribution, and then `.max()` was called -- which is pure argmax, completely ignoring the probability distribution. Temperature had been inert for every single run since the project started.

**Consequences.** The generator was always operating at temperature=0, picking the single most confident token at every step with no randomness. In blind mode, where the only input is a fully masked sequence of a given length, this meant every sequence of the same length was filled identically. Blind mode produced exactly one sequence per length bucket. Seeded mode accidentally produced diversity only because different seeds produced different logit distributions -- the 10% seed was doing all the diversity work.

During GAN training, the critic was consistently seeing the same small set of fake sequences repeating. It quickly memorized them, critic loss collapsed to near zero in a few steps, and no meaningful adversarial dynamic ever developed.

**Fix.** Replace `.max(dim=-1)` with `torch.multinomial()` so token selection is sampled from the probability distribution. Temperature then actually controls sampling diversity as intended.

---

### Bug 2 — Gradient Penalty Bug (loss.py)

**Discovery.** Identified during a code audit around the same time. In `compute_gradient_penalty()`, the interpolated embeddings were passed to the critic, but the critic's `dim==3` branch at the time skipped the ProtBERT transformer entirely and went straight to the classification head:

```python
elif input_data.dim() == 3:  # Embeddings
    last_hidden_state = input_data  # No transformer, no attention
```

The gradient penalty is supposed to enforce the Lipschitz constraint on the actual critic function. The actual critic function during training is: token IDs → full ProtBERT transformer (30 layers of attention) → classification head → score. The GP was enforcing the constraint on: raw interpolated embeddings → classification head → score. These are completely different functions. The theoretical guarantees of WGAN-GP (stable training, meaningful Wasserstein distance estimate) depend on the GP constraining the correct function. They were not.

A secondary issue in the same function was `attention_mask = torch.ones(interpolates.size()[:2])` treating all padding positions as real content during the GP computation.

**Fix.** Rewrite `compute_gradient_penalty()` to interpolate at the embedding layer and then pass through the full ProtBERT transformer before the classification head. Pass the actual attention masks from both real and fake sequences and compute a joint mask for the interpolated input. Zero out gradient contributions at padding positions before computing the gradient norm.

---

### Bug 3 — Generator Gradient Flow (Architectural Issue)

**Discovery.** Identified during the same code audit period. This is the most fundamental issue in the project and the one with the longest history.

The generator's `generate()` method returns `input_ids` -- a `torch.long` tensor of discrete integer token IDs. In the generator update step of both training scripts:

```python
fake_data = generate_fakes_for_batch(...)   # returns torch.long
fake_scores = critic(fake_data, ...)
g_loss = generator_loss(fake_scores)
g_loss.backward()
gen_optimizer.step()
```

When `backward()` runs, it traces the computation graph back through the critic's layers, but then hits the integer `input_ids` and stops. Integer indices are not differentiable -- you cannot ask "how much did choosing token 203 rather than token 202 affect the loss?" because 203 and 202 are not adjacent on a continuous scale. PyTorch cannot propagate gradient through a discrete lookup index.

The consequence is that `generator.protbert` receives a gradient of exactly zero on every generator update step. The only effect of `gen_optimizer.step()` is AdamW weight decay, which slowly erodes the fine-tuned weights. The generator was never trained adversarially. Every sequence ever generated came from the fine-tuned ProtBERT in whatever state it happened to be in -- GAN training epochs were irrelevant to generation quality.

This also explains why the best candidate (2508_len279) came from a GAN checkpoint but is structurally equivalent to what a fine-tuned ProtBERT would produce -- because it is exactly that.

**Why it was not caught earlier.** The argmax bug was masking this bug completely. With argmax, the critic was seeing a tiny fixed set of repeating fake sequences. It memorized them immediately, critic loss went to zero, and loss curves looked stable. The loss curves that would have revealed the gradient bug -- a generator that never improves, an oscillating critic, no adversarial dynamic -- were hidden behind the degenerate behavior caused by argmax. Even if gradients had been flowing, the argmax fakes were so repetitive that the gradient signal from a saturated critic would have been near zero anyway. The two bugs covered for each other.

**Fix.** Replace discrete integer passing with a soft embedding pass-through during generator update steps. Instead of calling `generate()`, take the raw float logits from ProtBERT, apply softmax to get a probability distribution, and compute a weighted average over the embedding matrix:

```python
logits = generator.protbert(input_ids=masked_input_ids, attention_mask=attention_mask).logits
soft_probs = F.softmax(logits / temperature, dim=-1)
embedding_matrix = critic.protbert.bert.embeddings.word_embeddings.weight
soft_embeds = soft_probs @ embedding_matrix
fake_scores = critic(soft_embeds, attention_mask=attention_mask)  # uses dim==3 branch
g_loss = generator_loss(fake_scores)
g_loss.backward()  # gradient flows all the way back to generator weights
```

This preserves a continuous differentiable path from the loss back to the generator's weights. The `generate()` method for actual sequence output is unchanged -- discrete token selection is still used when producing sequences for evaluation.

---

### Bugs 4-9 — Additional Issues Found in Code Audit (April 2026)

Beyond the three major bugs above, a Claude Code audit of the full codebase identified several additional issues:

**BUG 4 (CRITICAL)** -- `compute_gradient_penalty` called with wrong signature in both training scripts. `loss.py` was updated to accept `real_mask` and `fake_mask` parameters, but the calling code in `10p_train.py:328` and `fully_masked_train.py:317` was never updated. Will crash at runtime with `TypeError`.

**BUG 5 (CRITICAL)** -- `calculate_plddt_scores_and_save_pdb` returns a tuple `(avg_score, scores_list)` but both training scripts treat it as a scalar and log it directly to wandb. Every pLDDT metric in every training run was logged as a tuple, not a float. All training pLDDT metrics in W&B are garbage.

**BUG 6 (Significant)** -- `generate_fake_sequences` in `val_metrics.py` computes a random temperature but then never uses it, always passing `temperature=1.0` to the generator. Evaluation sequences during training always used temperature=1.0 regardless of the random sampling.

**BUG 7 (Significant)** -- `fully_masked_train.py` sets the attention mask during generation to exclude `[MASK]` tokens. Should exclude `[PAD]` tokens instead. Masking out `[MASK]` tokens prevents ProtBERT from attending to those positions as context, which is the opposite of what bidirectional attention needs.

**BUG 8 (Moderate)** -- NaN guard in `calculate_plddt_scores_and_save_pdb` filters out NaN values correctly, then immediately overwrites the result with an unfiltered average on the next line.

**BUG 9 (Quality of life)** -- `generate_fake_batch` in `fully_masked_train.py` defaults to `debug=True`, flooding SLURM logs with per-position mask counts on every training batch.

---

## Current State of Knowledge

After the bug audit, the honest summary of what the project knows:

**What the fine-tuned ProtBERT can do.** It can generate DNMT-like sequences that pass structural evaluation and, in 9 cases, survive the full AlphaFold3 co-folding pipeline with functional binding distances. The best candidate has been independently validated by a structural chemistry expert. This is a real result.

**What the GAN has done so far.** Nothing. The generator has never been adversarially trained. The critic trained successfully against a static, collapsed generator, but that training was useless because the critic's learned knowledge was never fed back into the generator.

**What the hyperparameter search found.** The best LR combinations and n_critic values were identified, but all of this search was conducted without a functioning GAN. The hyperparameter behavior may change after the gradient fix is applied.

**What the evaluation pipeline found.** Progres and scAccuracy are poor predictors of DNA-SAM binding proximity and should not be used as hard filters before AF3. The only reliable functional filter is AF3 co-folding with binding distance. The 7.25 Å threshold is appropriate.

---

## Infrastructure Notes

The project runs across three environments:

Local machine is used for development, code editing, and light testing.

Anzu is the Hacettepe BioDataSciLab GPU server (Ubuntu, multiple GPUs). Used for mid-scale evaluation runs, debugging, and as a staging point for MN5 transfers.

MareNostrum5 (MN5) is the BSC supercomputer with thousands of H100s. Used for large GAN training runs and AlphaFold3 evaluation. MN5 has no internet access -- files are uploaded via SCP and downloaded via SFTP. W&B runs are logged offline on MN5 then downloaded to Anzu and synced from there.

The Git workflow uses a two-remote setup: GitHub as origin, MN5 local repo as a second remote. Because MN5 cannot reach GitHub, changes are synced through local machine or Anzu as an intermediary.

---

## Planned Next Steps

In priority order, given the bugs described above:

1. Fix the generator gradient flow (soft embedding pass-through during generator updates).
2. Fix BUG 4 -- training scripts crash at runtime with wrong GP signature.
3. Fix BUG 5 -- unpack pLDDT return tuple so W&B logs actual float values.
4. Fix remaining bugs 6-9.
5. Add uniqueness ratio logging to `run_evaluation()` -- a one-line addition that would have caught the argmax bug on the first run.
6. Refactor shared code from both training scripts into `train_utils.py` so fixes only need to be applied once.
7. Verify blind mode diversity with a small generation test after the multinomial fix.
8. Re-run the 30 AF3 error sequences from the 120k batch.
9. Re-run 120k evaluation with scAccuracy filter removed.
10. Short 2-3 epoch training test on best hyperparameter config to assess whether gradient fix changes dynamics before committing to large-scale HPC runs.
11. Full new large-scale generation and evaluation campaign.
12. Evaluate Gökay's uniform top-k sampling suggestion (sample uniformly from top-k tokens rather than proportionally from the full distribution, inspired by EvoDiff, with optional adaptive k based on critic feedback).