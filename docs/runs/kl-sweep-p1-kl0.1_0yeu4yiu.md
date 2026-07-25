# kl-sweep-p1-kl0.1 (0yeu4yiu)

**Date:** 2026-07-25
**Script:** 10p_train.py
**wandb:** https://wandb.ai/ufuktanriverdi1-hacettepe-university/protgen-gan/runs/0yeu4yiu

## Config

| Param | Value |
|---|---|
| n_critic | 4 |
| lr_gen | 5e-6 |
| lr_critic | 5e-5 |
| lambda_kl | 0.1 |
| lambda_gp | 5 |
| temperature | 1.0 |
| n_epochs | 5 |
| batch_size | 8 |
| num_eval_sequences | 30 |
| max_train_seqs | not set (full dataset, ~52,637 seqs) |

(all other flags left at script defaults, per the sweep design doc)

## Final-epoch metrics

(final logged values, step 2635 / epoch 5)

| Metric | Value |
|---|---|
| kl_loss | 0.0409 |
| gen_grad_norm | 1.150 |
| unique_ratio | 1.0 |
| plddt | 0.7414 |
| scAccuracy | 0.3821 |
| progres | 0.9076 |
| pairwise_tm | 0.7366 |
| generator_loss | -0.1032 |
| critic_loss | 4.9998 |

## Per-epoch trend

Structural/eval metrics were logged once per epoch (evaluated on `num_eval_sequences=30`
generated sequences); values below are at each epoch's eval point (epochs 1-5, step
~527/1054/1581/2108/2635 respectively; an additional pre-epoch-1 baseline eval was logged
at step 2 and is included for reference):

- plddt: 0.756 (pre) → 0.746 → 0.762 → 0.759 → 0.757 → 0.741 (epochs 1-5)
- scAccuracy: 0.409 (pre) → 0.392 → 0.397 → 0.398 → 0.393 → 0.382 (epochs 1-5)
- progres: 0.915 (pre) → 0.911 → 0.917 → 0.913 → 0.906 → 0.908 (epochs 1-5)
- pairwise_tm: 0.795 (pre) → 0.789 → 0.795 → 0.832 → 0.806 → 0.737 (epochs 1-5)
- unique_ratio: 1.0 at every eval point (epochs 1-5) — no mode collapse.

`kl_loss` and `gen_grad_norm` are logged per-batch (not per-epoch); their behavior over
training:
- Epoch 1 (first-epoch freezing, both ProtBERT copies frozen): `gen_grad_norm = 0`
  throughout, as expected. `kl_loss` stayed small and stable, roughly 0.002-0.03.
- From epoch 2 on (unfreezing, ~step 530 onward): `gen_grad_norm` becomes consistently
  nonzero, typically in the 0.5-2.9 range per batch, with occasional larger spikes (e.g.
  6.18 at step 707, 22.47 at step 775) that did not recur or destabilize subsequent steps.
  `kl_loss` rose and became noisier, oscillating roughly in the 0.02-0.18 range for the
  remainder of training (epochs 2-5), ending at 0.041 at the final logged step. This is
  the highest lambda_kl value in the Phase 1 sweep (0.1), consistent with a larger anchor
  term than the lower-lambda_kl arms.

## Verdict

This run (lambda_kl=0.1, the largest value in the Phase 1 grid) completed all 5 epochs
without NaNs or unbounded blow-up. `gen_grad_norm` was zero during the frozen first epoch
as expected and became consistently nonzero from epoch 2 onward, confirming the generator
received adversarial gradient. `kl_loss` was bounded throughout (roughly 0.002-0.18, no
resemblance to the pre-fix 89-1236 oscillation range), though noticeably noisier and
higher on average than during the frozen epoch, with two isolated `gen_grad_norm` spikes
(6.18 and 22.47) that did not recur or cause subsequent instability. `unique_ratio` held
at 1.0 across every eval point, so there is no sign of mode collapse. Structural/quality
metrics were roughly flat to mildly declining over the 5 epochs: `plddt` 0.746→0.741,
`scAccuracy` 0.392→0.382, `progres` stayed close to ~0.91 throughout, and `pairwise_tm`
fluctuated (peaking at 0.832 in epoch 3) before ending lower at 0.737 in epoch 5. None of
these metrics show the sharp, monotonic epoch-over-epoch collapse seen in the
lambda_kl=0 reference run described in `docs/GENERATOR_GRADIENT_FIX.md` Stage 3.
