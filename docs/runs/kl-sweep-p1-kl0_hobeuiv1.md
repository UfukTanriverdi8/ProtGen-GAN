# kl-sweep-p1-kl0 (hobeuiv1)

**Date:** 2026-07-25
**Script:** 10p_train.py
**wandb:** https://wandb.ai/ufuktanriverdi1-hacettepe-university/protgen-gan/runs/hobeuiv1

## Config

| Param | Value |
|---|---|
| n_critic | 4 |
| lr_gen | 5e-6 |
| lr_critic | 5e-5 |
| lambda_kl | 0 |
| lambda_gp | 5 |
| temperature | 1.0 |
| n_epochs | 5 |
| max_train_seqs | not set (full dataset, ~52,637 seqs) |
| batch_size | 8 |
| wd_gen | 0.01 |
| wd_critic | 0.01 |
| eval_batch_size | 4 |
| num_eval_sequences | 30 |

Phase 1 arm of the lambda_kl sweep (`docs/sweeps/lambda-kl-sweep-2026-07.md`), lambda_kl=0 — i.e. no KL anchor term, clean adversarial-only generator loss.

## Final-epoch metrics

(step 1320, epoch 5)

| Metric | Value |
|---|---|
| kl_loss | not logged (lambda_kl=0, no KL anchor term computed) |
| gen_grad_norm | 1.84e-07 (last sampled point, step 1282) |
| unique_ratio | 1.0 |
| plddt_score | 0.6698 |
| scAccuracy | 0.4003 |
| progres | 0.9311 |
| pairwise_tm | 0.6689 |
| generator_loss | -0.3841 |
| critic_loss | 5.0000 |

## Per-epoch trend

Sampled at end of each epoch (steps 264, 528, 792, 1056, 1320 → epochs 1–5); step 1 is the pre-training baseline eval.

- progres: 0.9144 → 0.9332 → 0.9160 → 0.9085 → 0.9311 (epochs 1-5; baseline 0.9108) — flat/noisy, no clear trend
- scAccuracy: 0.3959 → 0.3997 → 0.3890 → 0.3821 → 0.4003 (epochs 1-5; baseline 0.3899) — flat/noisy
- pairwise_tm: 0.7724 → 0.7069 → 0.6642 → 0.6141 → 0.6689 (epochs 1-5; baseline 0.7878) — declines through epoch 4, partial rebound at epoch 5, net decline over the run (diversity of generated structures dropping)
- plddt_score: 0.7554 → 0.6704 → 0.6657 → 0.6596 → 0.6698 (epochs 1-5; baseline 0.7545) — drops sharply after epoch 1, then flat/slightly declining through epoch 4, small rebound at epoch 5
- unique_ratio: 1.0 → 1.0 → 1.0 → 1.0 → 1.0 (epochs 1-5) — no mode collapse by this metric
- generator_loss: -0.0287 → -0.1739 → -0.3871 → -0.3855 → -0.3841 (epochs 1-5; baseline -0.0713) — drops then plateaus from epoch 3 on
- critic_loss: 5.246 → 5.064 → 5.000 → 5.000 → 5.000 (epochs 1-5; baseline 4.936) — converges to and sits exactly at 5.000 from epoch 3 onward
- gen_grad_norm: near-zero (0) through early epoch 2 (steps ≤235), spikes to 0.02–0.41 across steps 298–494 (mid epoch 2), then collapses to ~1e-7 order for the remainder of training (steps 612–1282, spanning epochs 3-5)

## Verdict

This run (lambda_kl=0, no KL anchor) held `unique_ratio` at 1.0 throughout all 5 epochs — no mode collapse by that metric. However, `pairwise_tm` (diversity) declined from 0.79 at baseline to a low of 0.61 by epoch 4 before a partial rebound to 0.67 at epoch 5, and `plddt_score` dropped from 0.75 at baseline to the 0.66–0.67 range by epoch 2 and stayed there, never recovering to baseline. `progres` and `scAccuracy` stayed flat/noisy across epochs with no clear directional trend. The most notable anomaly is `gen_grad_norm`: it briefly spiked as high as 0.41 during mid-epoch 2 (steps 298-494) but then collapsed to the ~1e-7 order of magnitude for the rest of training (epochs 3-5), coinciding with `critic_loss` converging to and sitting exactly at 5.000 from epoch 3 onward — consistent with the critic saturating and the generator receiving effectively no adversarial gradient signal for the majority of the run. `generator_loss` correspondingly plateaus at -0.385 from epoch 3 on. No NaN values were observed in the sampled history.
