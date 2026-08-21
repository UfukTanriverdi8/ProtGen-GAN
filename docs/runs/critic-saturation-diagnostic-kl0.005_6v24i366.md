# critic-saturation-diagnostic-kl0.005 (6v24i366)

**Date:** 2026-08-21
**Script:** 10p_train.py
**wandb:** https://wandb.ai/ufuktanriverdi1-hacettepe-university/protgen-gan/runs/6v24i366

## Config

Confirmed via `query_wandb_tool` GraphQL (the run's actual `config` field, not just `_wandb`
client metadata — see CLAUDE.md item 21).

| Param | Value |
|---|---|
| seed | 89 |
| n_critic | 4 |
| lr_gen | 0.000005 (5e-6) |
| lr_critic | 0.00005 (5e-5) |
| wd_gen | 0.01 |
| wd_critic | 0.01 |
| lambda_gp | 5 |
| lambda_kl | 0.005 |
| temperature | 1 |
| n_epochs | 15 |
| batch_size | 8 |
| eval_batch_size | 4 |
| num_eval_sequences | 30 |
| holdout_size | 200 |
| loss_log_every | 25 |
| holdout_eval_fakes | false |

`holdout_eval_fakes: false` — confirmed `holdout_fake_score_mean/std` never fired for this
run (queried directly, zero rows returned at any sampling density).

This is the `lambda_kl=0.005` arm of the critic-saturation diagnostic (CLAUDE.md TODO item
20), re-running the lambda_kl sweep's Phase 2 config with held-out critic validation tooling
enabled. `max_train_seqs` was not passed (full ~52k-sequence pool minus the 200-sequence
holdout).

## Final-epoch metrics

Values at the last logged step (`_step` 8070/8071, epoch 14, tag `end`/final `step`).

| Metric | Value |
|---|---|
| critic_loss | 5.0076 |
| generator_loss | 5.9373 |
| kl_loss | 0.7935 (last densely-logged per-step value, step 8041) |
| gen_grad_norm | 0.1675 |
| unique_ratio | 1.0 |
| plddt_score | 0.7137 |
| scAccuracy | 0.4026 |
| progres | 0.9065 |
| pairwise_tm | 0.7580 |
| holdout_real_score_mean | -5.8571 |
| holdout_real_score_std | 0.002626 |
| train_real_score_mean | -5.8561 |
| train_real_score_std | 0.001780 |
| train_fake_score_mean | -5.8560 |
| train_fake_score_std | 0.001767 |
| critic_loss_step (last `step`-tag row) | 4.9673 |

`holdout_fake_score_mean/std`: not logged (holdout_eval_fakes=false, as expected).

## Per-epoch trend

Structural/adversarial metrics below are the 16 baseline+per-epoch checkpoints (baseline
pre-training, then epochs 0–14; `_step` ≈ 3, 538, 1076, 1614, 2152, 2690, 3228, 3766, 4304,
4842, 5380, 5918, 6456, 6994, 7532, 8070):

- **critic_loss** (baseline → epoch14): 4.970 → 4.975 → **2782.28** (epoch1, single massive
  spike) → 5.000 → 5.000 → 5.000 → 5.000 → 4.9999 → 4.9998 → 4.9995 → 4.9984 → 4.9926 →
  4.9258 (epoch12) → 5.2932 (epoch13) → 5.0872 (epoch14, partial) → 5.0076 (final). Pins at
  essentially exactly `5.0000` from epoch2 through epoch11, then starts drifting away from
  5.0 (down to 4.93, then up to 5.29) right around epoch12–13, coinciding with the score-std
  and gradient events below.
- **train_real_score_std** (dense, every 25 batches): ~0.001–0.05 during the first ~540 steps
  (epoch0, still warming up), collapses to ~2e-6–1e-5 from step ~642 through step ~5800
  (epoch1 through epoch~10, i.e. **10+ epochs of near-zero variance while critic_loss sits at
  5.0000**), then starts climbing from step 5842 (0.0027) through a spike at step 6458
  (0.0802, ~epoch12), and oscillates in the 0.001–0.03 range for the remainder, ending at
  0.0018 (step 8045).
- **holdout_real_score_std** (tag `mid`/`end`, held-out real sequences never trained on):
  same shape — 3–6e-6 from step 810 through step 5919 (epoch~1–11), then rises: 0.0007
  (5919) → 0.0021 (6190) → **0.0638 spike** (6457, ~epoch12) → 0.0107 (6728) → oscillates
  0.0002–0.0158 → 0.0026 (final, step 8071). The holdout curve tracks the training-pool
  curve closely — the std collapse (and later recovery) is not a training-data-memorization
  artifact, it shows up on sequences the critic never saw.
- **gen_grad_norm**: exactly 0 for the first ~500 steps (epoch0, first-epoch freezing as
  designed), jumps to 3.5–9.9 at step 544–570 (unfreeze), settles to a small, stable 0.02–0.13
  band through step ~6140 (epoch0 unfreeze through epoch~11), then spikes hard starting step
  6310 (3.63) escalating to 17.8 (6672), **74.2 (6738)**, **80.8 (6820)** — roughly epoch
  12–13 — before dropping back into a noisier-than-early 0.1–0.8 band (with an occasional
  spike, e.g. 4.67 at step 7292) through the end, final value 0.1675.
- **kl_loss**: ~0.003–0.01 during the frozen epoch0, jumps to 0.05–0.13 right after unfreeze,
  stays in a 0.01–0.1 band through roughly epoch0–9 (step ~5000), begins rising from step 5497
  (0.146) through step 6141 (0.35), then escalates sharply from step 6309 (1.08) to a peak of
  3.79 (step 6763, ~epoch12), before gradually declining back down to 0.79 by the final step
  (8041) — still well above the early-training band but trending down by the end.
- **plddt_score**: 0.739 → 0.744 → 0.762 → 0.762 → 0.746 → 0.722 → 0.754 → 0.757 → 0.756 →
  0.752 → 0.766 → 0.739 → 0.734 → 0.762 → 0.756 → 0.714 (baseline→epoch14). Flat/noisy
  throughout, no clear trend tied to the epoch12–13 gradient/score-std event.
- **scAccuracy**: 0.400 → 0.391 → 0.397 → 0.395 → 0.394 → 0.377 → 0.376 → 0.395 → 0.399 →
  0.390 → 0.393 → 0.403 → 0.339 (epoch12 dip) → 0.399 → 0.398 → 0.403. Also flat/noisy.
- **progres**: 0.913 → 0.896 → 0.917 → 0.917 → 0.909 → 0.877 → 0.917 → 0.919 → 0.920 → 0.906
  → 0.915 → 0.920 → 0.927 → 0.918 → 0.908 → 0.906. Flat/noisy, ~0.88–0.93 band throughout.
- **pairwise_tm**: 0.790 → 0.764 → 0.777 → 0.775 → 0.756 → 0.704 → 0.705 → 0.810 → 0.812 →
  0.758 → 0.766 → 0.762 → 0.773 → 0.751 → 0.722 → 0.758. Flat/noisy.
- **unique_ratio**: 1.0 at every logged checkpoint — no mode collapse in generated sequences
  by this metric across the whole run.
- **generator_loss**: -0.015 → -0.075 → -0.016 → -0.013 → -0.013 → -0.013 → -0.013 → -0.014
  → -0.020 → -0.035 → -0.056 → -0.073 → **-0.770 (epoch12)** → **-1.644 (epoch13)** →
  **+3.862 (epoch14, partial)** → **+5.937 (final)**. Stable near-zero for most of training,
  then a sharp regime change starting epoch12 that includes a sign flip by the final two
  evaluation points — coincides exactly with the gradient/score-std event above.

## Verdict

`critic_loss` does saturate at essentially exactly `5.0000` for this run, but not for the
whole run: it pins tightly from epoch2 through roughly epoch11 (10 epochs), and during that
entire window both `train_real_score_std` and `holdout_real_score_std` collapse to ~1e-6–1e-5
— several orders of magnitude below the pre-saturation and post-saturation values — matching
the degenerate-collapse signature described in CLAUDE.md item 20 (critic output stopped being
a function of its input) rather than genuine convergence. The held-out real-sequence std
tracks the training-pool std closely through this period, so the collapse is not an artifact
of the critic memorizing training reals. Starting around epoch12 (step ~6300–6900), the run
exits this collapsed state on its own: `critic_loss` drifts off 5.0000 (down to 4.93, then up
to 5.29), both score-stds spike back up by 1–2 orders of magnitude, `gen_grad_norm` spikes
dramatically (peaking at 74–81, ~1000x its collapsed-period band), `kl_loss` rises from ~0.1
to a peak of 3.8, and `generator_loss` undergoes a sign flip and blows up to +5.9 by the final
step. One additional anomaly not part of the collapse/recovery narrative: a single massive
`critic_loss` spike to 2782 at the epoch1 checkpoint, immediately after first-epoch freezing
ends — a one-step transient, fully resolved by the next checkpoint, that does not recur.
Structural quality metrics (plddt_score, scAccuracy, progres, pairwise_tm, unique_ratio) stay
flat and noisy throughout, showing no clear relationship to either the saturation period or
the epoch12+ recovery/blowup — consistent with CLAUDE.md's existing finding that these
metrics plateau early regardless of what the adversarial dynamics are doing.
