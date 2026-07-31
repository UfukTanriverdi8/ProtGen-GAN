# kl-sweep-p2-retry-kl0.005 (w8c2wl85)

**Date:** 2026-07-31 (wandb `createdAt` 2026-07-31T06:51:33Z)
**Script:** 10p_train.py
**wandb:** https://wandb.ai/ufuktanriverdi1-hacettepe-university/protgen-gan/runs/w8c2wl85

## Config

Both `run.config` and `run._attrs['rawconfig']` were queried directly via the wandb Python
API (bypassing MCP entirely, since the wandb MCP server is currently unreachable from this
network due to a DNS resolution failure on `mcp.withwandb.com`). Both returned **empty** for
real hyperparameters — only `_wandb` client/framework metadata (`cli_version: 0.28.0`,
`python_version: 3.12.13`, `huggingface_version: 4.46.3`, `framework: huggingface`, etc.),
no `lambda_kl`, `lr_gen`, `n_critic`, or any other run-specific key. This directly
contradicts `docs/sweeps/lambda-kl-sweep-2026-07.md`'s "Correction (2026-07-25)" note, which
claimed the earlier empty-config problem was purely an MCP-tool limitation and that the
Python API/raw `config.yaml` on disk confirmed the real values. That claim was not
independently re-verified here, and this run's data shows the same gap through the Python
API — so the gap should not be assumed to be an MCP artifact.

The values below are taken instead from `slurms/sweeps/lambda_kl_sweep_p2.sh` (the actual
SLURM submission script used to launch this run, array task 1 of `kl_list=(0.005 0.05)`),
which are known with certainty since that script directly sets these CLI flags — but this is
**not independently confirmed by wandb's own config field** via any access method tried so
far.

| Param | Value |
|---|---|
| n_critic | 4 |
| lambda_gp | 5 |
| lr_gen | 5e-6 |
| lr_critic | 5e-5 |
| lambda_kl | 0.005 |
| temperature | 1.0 |
| n_epochs | 15 |
| batch_size | 8 |
| num_eval_sequences | 30 |
| max_train_seqs | full dataset (not capped) |
| mode | seeded (10p_train.py) |

wandb tags: `kl-sweep-p2`, `kl-sweep-p2-retry`.

## Final-epoch metrics

Final logged values, epoch 15 (`_step` 7905, summary `tag: "end"`):

| Metric | Value |
|---|---|
| kl_loss | 0.5300 |
| gen_grad_norm | 0.2219 |
| unique_ratio | 1.0 |
| plddt_score | 0.7381 |
| scAccuracy | 0.3863 |
| progres | 0.9004 |
| pairwise_tm | 0.7278 |
| critic_loss | 4.9967 |
| generator_loss | -0.0382 |

## Per-epoch trend

All 15 epoch-end rows (`tag: "end"`) confirmed present in `scan_history()` (7906 total
logged rows, run state `finished`).

- plddt_score: 0.764 → 0.745 → 0.770 → 0.746 → 0.774 → 0.731 → 0.724 → 0.770 → 0.751 → 0.742 → 0.764 → 0.746 → 0.773 → 0.774 → 0.738 (epochs 1-15). Oscillates in a 0.724-0.774 band, no sustained trend — similar overall shape to the original run's 0.729-0.758 band, slightly wider.
- scAccuracy: 0.408 → 0.387 → 0.389 → 0.386 → 0.402 → 0.317 → 0.359 → 0.369 → 0.353 → 0.377 → 0.388 → 0.388 → 0.399 → 0.402 → 0.386 (epochs 1-15). Mostly 0.35-0.41, with a low dip to 0.317 at epoch 6 — flat/noisy overall, comparable to the original's 0.38-0.40 band but with more variance.
- progres: 0.907 → 0.915 → 0.909 → 0.900 → 0.911 → 0.907 → 0.895 → 0.921 → 0.922 → 0.888 → 0.903 → 0.913 → 0.918 → 0.898 → 0.900 (epochs 1-15). Stable ~0.89-0.92, matching the original run's range closely.
- pairwise_tm: 0.819 → 0.753 → 0.790 → 0.775 → 0.837 → 0.657 → 0.691 → 0.809 → 0.808 → 0.750 → 0.752 → 0.767 → 0.815 → 0.776 → 0.728 (epochs 1-15). Wider swings than the original (0.657-0.837 vs original's 0.704-0.798), but similarly no sustained decline — ends comparable to where it started.
- unique_ratio: 1.0 at every epoch (1-15). No mode collapse, matching the original run.
- kl_loss (per-batch, 3945 logged points): epoch-bucketed min/median/max (steps//527+1):
  epoch 1: 0.001/0.004/0.02 → epoch 2: 0.005/0.21/1.52 → epoch 3: 0.17/0.84/2.28 →
  epoch 4: 0.08/0.25/0.79 → epoch 5: 0.09/0.53/2.77 → epoch 6: 0.39/1.83/4.81 →
  epoch 7: 0.80/1.92/5.10 → epoch 8: 1.12/2.34/4.06 → epoch 9: 1.40/2.80/4.58 →
  epoch 10: 1.64/3.16/6.83 (peak) → epoch 11: 0.90/1.83/2.89 → epoch 12: 0.68/1.80/3.04 →
  epoch 13: 0.61/1.41/2.34 → epoch 14: 0.48/0.96/1.63 → epoch 15: 0.27/0.67/1.09 (ends at
  0.530, the epoch-15 summary value). This is a materially different trajectory than the
  original run: the original stayed low (~0.004-0.03) through the first third, then settled
  into a 0.02-0.65 band for the back two-thirds. This retry starts similarly low in epoch 1
  but climbs steadily to a median of ~2.3-3.2 across epochs 8-10 before declining again in
  epochs 11-15 — overall much noisier and higher-magnitude than the original, though it does
  not diverge (comes back down by epoch 15).
- gen_grad_norm (per-batch, 3945 logged points): 0 throughout epoch 1 (first-epoch freezing,
  same as original), turns nonzero from step 529 (early epoch 2) onward. Epoch-bucketed
  min/median/max: epoch 2: 0.07/1.26/109.3 → epoch 3: 0.10/0.57/68.6 → epoch 4: 0.07/0.15/73.0
  → epoch 5: 0.07/1.97/108.4 → epoch 6: 0.30/36.5/380.5 → epoch 7: 0.21/17.0/136.6 →
  epoch 8: 8.16/50.1/157.98 → epoch 9: 32.3/103.9/510.6 (peak) → epoch 10: 0.15/27.6/233.4 →
  epoch 11: 0.12/0.20/46.0 → epoch 12: 0.12/33.2/113.2 → epoch 13: 0.08/0.19/12.0 →
  epoch 14: 0.10/0.18/0.50 → epoch 15: 0.08/0.17/1.44 (ends at 0.222, the epoch-15 summary
  value). 1,652 of the 3,945 logged points exceed 5.0, with genuine large spikes up to
  510.6 (epoch 9) — this is dramatically different from the original run, which stayed
  bounded in a ~0.03-0.25 range for most of training with only a handful of small transient
  spikes (max observed ~11.9). By epochs 14-15 this retry's gen_grad_norm settles back down
  into a range comparable to the original's steady-state band, but epochs 6-12 show
  sustained, large-magnitude instability that the original run never exhibited.
- critic_loss (16 logged points, epoch 1 logged twice — once early at step 2, once at
  step 527): 4.86 (epoch 1, early) → 75.87 (epoch 1, late) → 7.23 (epoch 2) → 3.55 (epoch 3)
  → 4.91 (epoch 4) → 3.73 (epoch 5) → 1.68 (epoch 6) → 2.00 (epoch 7) → 0.15 (epoch 8) →
  -0.01 (epoch 9) → 1.10 (epoch 10) → 4.28 (epoch 11) → 0.92 (epoch 12) → 4.98 (epoch 13) →
  5.00 (epoch 14) → 4.997 (epoch 15). Never pins persistently at exactly 5.0000 the way the
  original run's critic_loss did from epoch 5 onward — this retry stays volatile across all
  15 epochs, only landing close to 5.0 transiently at epochs 13-15 without settling to the
  same repeated-decimal value.
- generator_loss (16 logged points, same double-epoch-1 pattern): 0.069 → 0.087 → 0.250 →
  0.334 → 0.137 → 0.483 → 1.826 → 2.389 → 4.425 → 4.791 → 5.810 → 10.056 (peak, epoch 11) →
  3.296 → -0.284 → -0.756 → -0.038 (epoch 15). Climbs to a much larger magnitude (peak
  10.06 at epoch 11) than the original run, which settled to a tight, stable ~-0.21 plateau
  from epoch 5 onward — this retry shows no comparable plateau and swings from strongly
  positive to negative across the back half of training.

## Verdict

Quality metrics (plddt_score, scAccuracy, progres, pairwise_tm, unique_ratio) tell largely
the same story as the original, temperature-buggy Phase 2 run: all stay flat/oscillating
across the full 15-epoch run with no sustained directional trend, unique_ratio holds at 1.0
throughout (no mode collapse), and the final-epoch values are close to the original's
(plddt 0.738 vs 0.747, scAccuracy 0.386 vs 0.386, progres 0.900 vs 0.911, pairwise_tm 0.728
vs 0.766). So fixing the temperature-pinning bug did not meaningfully change the picture on
generation quality for lambda_kl=0.005 — the ranges are comparable, if anything slightly
wider/noisier in this retry, which is consistent with expected run-to-run variance rather
than a systematic shift attributable to the fix.

The training-dynamics side is a different story, and this is the important finding to flag
explicitly. The original run's `critic_loss` pinned at exactly 5.0000 (to several decimal
places) for every epoch from epoch 5 through epoch 15 — a clean, textbook saturation
signature — with `generator_loss` correspondingly settling to a tight, stable ~-0.21
plateau over the same span. This retry, run with the identical hyperparameters (n_critic=4,
lr_gen=5e-6, lr_critic=5e-5, lambda_kl=0.005, temperature=1.0, 15 epochs), shows neither
pattern: `critic_loss` stays volatile across all 15 epochs (75.9, 7.2, 3.6, 4.9, 3.7, 1.7,
2.0, 0.15, -0.01, 1.1, 4.3, 0.9, then only approaching ~5.0 transiently at epochs 13-15
without locking to the repeated-decimal value), and `generator_loss` swings as high as
+10.06 (epoch 11) instead of settling into any plateau. `gen_grad_norm` is correspondingly
far noisier here too — sustained large spikes up to 510.6 through epochs 6-12, versus the
original's tightly bounded ~0.03-0.25 range with only a couple of small transient outliers.
`kl_loss` likewise climbs to a much higher band (median ~2.3-3.2 across epochs 8-10) before
coming back down by epoch 15, versus the original's comparatively contained 0.02-0.65 band.

Because both runs used identical hyperparameters and only the temperature-pinning bug fix
and normal stochastic variation (batch order, masking positions, multinomial sampling draws)
differ between them, this divergence is most plausibly attributable to run-to-run randomness
in the adversarial training dynamics rather than to the temperature fix itself — the
temperature fix only affects `generate_fake_sequences` (the evaluation-time sequence
generation path used for plddt/scAccuracy/progres/pairwise_tm/unique_ratio), not the
training-loop generation path that produces `critic_loss`/`generator_loss`/`gen_grad_norm`/
`kl_loss` (that path already used the pinned temperature correctly in both the original and
this retry run, per the CLAUDE.md fix history). In other words: the fact that this retry's
critic never saturates the same way, while the original's did, is evidence that critic
saturation under lambda_kl=0.005 is not a deterministic function of the hyperparameter
setting alone — it can apparently be avoided or triggered by run-to-run randomness with
these same fixed hyperparameters. This weakens confidence in attributing the earlier
critic-saturation narrative (and by extension any lambda_kl-vs-lambda_kl comparison built on
top of it, e.g. item 20 in CLAUDE.md's TODO) purely to lambda_kl choice, and argues for
treating single-seed sweep runs as noisy signals rather than clean hyperparameter
comparisons — multiple seeds per arm would be needed to separate the two effects with
confidence.
