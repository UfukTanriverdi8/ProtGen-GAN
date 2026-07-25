# kl-sweep-p1-kl0.01 (kwm7j52c)

**Date:** 2026-07-25
**Script:** 10p_train.py
**wandb:** https://wandb.ai/ufuktanriverdi1-hacettepe-university/protgen-gan/runs/kwm7j52c

## Config

wandb's logged `config` blob for this run contains only `_wandb` client metadata
(framework/version info) — no hyperparameters were captured in `wandb.config` for
this run. Values below are taken from the run name and the sweep design doc
(`docs/sweeps/lambda-kl-sweep-2026-07.md`, "Fixed hyperparameters" table), which
applied identically to all 5 Phase 1 arms.

| Param | Value |
|---|---|
| n_critic | 4 |
| lr_gen | 5e-6 |
| lr_critic | 5e-5 |
| lambda_kl | 0.01 (from run name; not confirmed in wandb config) |
| temperature | 1.0 |
| n_epochs | 5 |
| max_train_seqs | None (full dataset, ~52,637 seqs) |
| num_eval_sequences | 30 |

(lambda_gp, wd_gen, wd_critic, batch_size, iteration_fill_rate: script defaults per design doc, not overridden)

## Final-epoch metrics

(epoch 5 / final logged step, `_step` 2635)

| Metric | Value |
|---|---|
| kl_loss | 0.0763 |
| gen_grad_norm | 0.1786 |
| unique_ratio | 1.0 |
| plddt_score | 0.7417 |
| scAccuracy | 0.3796 |
| progres | 0.8963 |
| pairwise_tm | 0.7136 |
| generator_loss | -0.0836 |
| critic_loss | 5.0027 |

## Per-epoch trend

(one eval point logged per epoch, epochs 1-5)

- kl_loss (per-step, sparse): ranged roughly 0.001 (early epoch 1) up to a peak of 0.54 (mid-run, step ~1213), settling around 0.05-0.17 by the end — bounded oscillation, no runaway growth.
- gen_grad_norm (per-step, sparse): 0 early in epoch 1 (frozen first epoch), then jumps to the 0.08-0.5 range from epoch 2 onward, with one outlier spike to 4.33 at step 1118; otherwise stable, no collapse to zero after epoch 1.
- unique_ratio: 1.0 → 1.0 → 1.0 → 1.0 → 1.0 (epochs 1-5) — no mode collapse.
- plddt_score: 0.7512 → 0.7534 → 0.7502 → 0.7870 → 0.7502 (epochs 1-5) — mild fluctuation, epoch 3 peak, ends essentially flat vs. epoch 1.
- scAccuracy: 0.3762 → 0.3855 → 0.3882 → 0.4139 → 0.3796 (epochs 1-5) — rises through epoch 4 then dips at epoch 5, ends slightly above epoch-1 start.
- progres: 0.9150 → 0.9149 → 0.8987 → 0.9146 → 0.8963 (epochs 1-5) — mild dip at epoch 2, recovers epoch 3-4, dips again epoch 5; net small decline from start.
- pairwise_tm: 0.7697 → 0.7539 → 0.7320 → 0.8365 → 0.7136 (epochs 1-5) — non-monotonic, epoch 3 low, epoch 4 peak, epoch 5 lowest of the run.

## Verdict

This run (lambda_kl=0.01, n_critic=4, seeded mode, 5 epochs, full dataset) held
`unique_ratio=1.0` throughout — no mode collapse. `gen_grad_norm` was zero during
the frozen first epoch as expected, then positive and roughly stable (0.08-0.5)
from epoch 2 on, with a single unexplained spike to 4.33 at step 1118 that did not
recur or destabilize subsequent steps. `kl_loss` stayed bounded (peak 0.54,
settling to 0.05-0.17), consistent with the post-fix behavior described in
CLAUDE.md and far from the pre-fix 89-1236 oscillation range. The four structural
quality metrics (plddt_score, scAccuracy, progres, pairwise_tm) did not show a
clean monotonic trend in either direction over 5 epochs — each fluctuates within
a narrow band (plddt_score 0.75-0.79, progres 0.90-0.92, pairwise_tm 0.71-0.84,
scAccuracy 0.38-0.41), with epoch 5 values close to or slightly below epoch 1 for
most metrics except scAccuracy. No NaNs or divergence were observed in the logged
history. Note: this run's wandb `config` object did not capture hyperparameters
(only client metadata was logged), so the lambda_kl=0.01 value and other config
fields above are inferred from the run name and sweep design doc rather than
confirmed directly from wandb config — a wandb.config logging gap that will
affect other Phase 1 run docs equally.
