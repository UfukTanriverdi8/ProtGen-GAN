# kl-sweep-p1-kl0.05 (v7ha9aab)

**Date:** 2026-07-25
**Script:** 10p_train.py
**wandb:** https://wandb.ai/ufuktanriverdi1-hacettepe-university/protgen-gan/runs/v7ha9aab

## Config

wandb's logged `config` blob for this run contains only the `_wandb` telemetry
key (framework/version metadata) — no hyperparameters were captured in
`run.config` for this run, so the table below is populated from the sweep's
design doc (`docs/sweeps/lambda-kl-sweep-2026-07.md`, "Fixed hyperparameters"
+ "Phase 1" sections) and the run name, not from a wandb config read.

| Param | Value |
|---|---|
| n_critic | 4 |
| lr_gen | 5e-6 |
| lr_critic | 5e-5 |
| lambda_kl | 0.05 |
| temperature | 1.0 |
| n_epochs | 5 |
| max_train_seqs | None (full dataset, ~52,637 seqs) |
| num_eval_sequences | 30 |
| lambda_gp / wd_gen / wd_critic / batch_size / iteration_fill_rate | 5.0 / 0.01 / 0.01 / 8 / 0.1 (script defaults, unchanged) |

## Final-epoch metrics

(epoch 5 / step 2635, from wandb summary)

| Metric | Value |
|---|---|
| kl_loss | 0.0733 |
| gen_grad_norm | 0.5598 |
| unique_ratio | 1.0 |
| plddt | 0.7422 |
| scAccuracy | 0.3934 |
| progres | 0.9007 |
| pairwise_tm | 0.7355 |
| generator_loss | 0.00634 |
| critic_loss | 4.9697 |

## Per-epoch trend

Eval metrics logged once per epoch (steps 527, 1054, 1581, 2108, 2635 for
epochs 1-5; an additional baseline point was logged at step 2, before epoch 1
training):

- plddt: 0.7670 (step-2 baseline) → 0.7536 → 0.7426 → 0.7488 → 0.7531 → 0.7422 (epochs 1-5)
- scAccuracy: 0.3895 → 0.3931 → 0.3859 → 0.4031 → 0.3930 → 0.3934
- progres: 0.9001 → 0.9189 → 0.8995 → 0.9044 → 0.9236 → 0.9007
- pairwise_tm: 0.7575 → 0.8066 → 0.7654 → 0.7861 → 0.7965 → 0.7355
- unique_ratio: 1.0 at every logged epoch (no mode collapse observed)
- generator_loss (epoch boundaries): 0.0662 → 0.0896 → 0.0937 → 0.0872 → 0.0955 → 0.0063
- critic_loss (epoch boundaries): 4.85 → 5.84 → 4.99 → 4.99 → 5.00 → 4.97
- gen_grad_norm: 0 for all of epoch 1 (backbone frozen per first-epoch-freezing design), then jumps once the generator backbone unfreezes at the start of epoch 2 (step 533), with large transient spikes immediately after unfreezing (344.5 at step 541, 25.4 at step 543), settling into a roughly 0.3-2.0 range for the rest of training with occasional isolated spikes (15.3 at step 741, 8.5 at step 1013, 30.2 at step 2330, 4.5 at step 2628)
- kl_loss: near-zero through epoch 1 (~0.002-0.01, consistent with the frozen generator producing near-identical logits to the frozen reference), rises after unfreezing and continues climbing across epochs 2-5, oscillating in a bounded band that widens over time (roughly 0.02-0.09 in epoch 2, up to 0.05-0.25 by epoch 5, with one isolated outlier of 0.4515 at step 2141); ends epoch 5 at 0.0733 per the summary

## Verdict

This run (lambda_kl=0.05) completed all 5 epochs without NaNs or the kind of
unbounded kl_loss blowup seen pre-fix (89-1236); kl_loss instead grew
gradually from near-zero (frozen epoch 1) into a widening but bounded
oscillation band (roughly up to 0.25, one outlier at 0.45) by epoch 5, and
gen_grad_norm was 0 during the frozen first epoch as expected, then positive
and mostly bounded (0.3-2.0) after unfreezing, with a few transient spikes
(largest 344.5 immediately after unfreezing, and isolated spikes up to ~30
later). unique_ratio held at 1.0 at every logged epoch, so no mode collapse.
Structural/quality metrics (plddt, scAccuracy, progres, pairwise_tm) did not
show a clear monotonic trend over the 5 epochs — plddt and pairwise_tm dipped
from their step-2 baseline and fluctuated epoch to epoch without a clean
decline or improvement, ending epoch 5 at plddt 0.742, scAccuracy 0.393,
progres 0.901, and pairwise_tm 0.735 (pairwise_tm's epoch-5 value is its
lowest of the run, after peaking at 0.797 in epoch 4). One anomaly worth
flagging: generator_loss drops sharply at the very last logged step (2635,
0.0063) versus the 0.087-0.096 range at every other epoch boundary, while
gen_grad_norm and kl_loss at that same step look unremarkable relative to
nearby steps — could be a single noisy final-batch value rather than a real
shift, but is called out here for the synthesis step to weigh.
