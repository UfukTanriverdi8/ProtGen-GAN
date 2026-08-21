# critic-saturation-diagnostic-kl0.05 (19y835y0)

**Date:** 2026-08-21
**Script:** 10p_train.py
**wandb:** https://wandb.ai/ufuktanriverdi1-hacettepe-university/protgen-gan/runs/19y835y0

## Config

Config confirmed via `query_wandb_tool` GraphQL (the direct `run.config` field, not just
`_wandb` metadata — see CLAUDE.md item 21 for background on that gap).

| Param | Value |
|---|---|
| n_critic | 4 |
| lr_gen | 5e-6 |
| lr_critic | 5e-5 |
| lambda_kl | 0.05 |
| lambda_gp | 5 |
| temperature | 1.0 |
| n_epochs | 15 |
| batch_size | 8 |
| eval_batch_size | 4 |
| wd_gen / wd_critic | 0.01 / 0.01 |
| seed | 89 |
| holdout_size | 200 |
| holdout_eval_fakes | **false** — no `holdout_fake_score_*` metrics were logged this run |
| loss_log_every | 25 |
| num_eval_sequences | 30 |

This is the `lambda_kl=0.05` arm of the critic-saturation diagnostic (CLAUDE.md TODO item 20),
re-running the lambda_kl sweep's Phase 2 config with the newly-added held-out critic
validation tooling (`--holdout_size`, `--loss_log_every`) enabled.

## Final-epoch metrics

(epoch 15, `tag=end`, step ~8070/8071)

| Metric | Value |
|---|---|
| critic_loss | 4.9993 |
| generator_loss | -0.0625 |
| gen_grad_norm | ~0.70 (last sampled point, step 8067) |
| kl_loss | ~0.086 (last sampled point, step 8066) |
| unique_ratio | 1.0 |
| plddt_score | 0.7659 |
| scAccuracy | 0.3992 |
| progres | 0.8965 |
| pairwise_tm | 0.8085 |
| holdout_real_score_mean | 0.0601 |
| holdout_real_score_std | 4.85e-05 |
| train_real_score_mean (last `step`-tag point, step 8045) | 0.0619 |
| train_real_score_std (last `step`-tag point, step 8045) | 5.75e-05 |
| holdout_fake_score_mean/std | not logged — `--holdout_eval_fakes` was not set for this run |

## Per-epoch trend

**critic_loss** (`baseline`, then `end`-tag per epoch, steps 3/538/1076/.../8070):
4.975 (baseline) → 4.975 (ep1) → **165.71 (ep2, anomalous spike)** → 5.086 (ep3) → 5.517 (ep4)
→ 4.99995 (ep5) → 4.99995 (ep6) → 4.99995 (ep7) → 4.99995 (ep8) → 4.99995 (ep9) →
4.99995 (ep10) → 4.99994 (ep11) → 4.99993 (ep12) → 4.99989 (ep13) → 4.99981 (ep14) →
4.99932 (ep15, final). From epoch 5 onward `critic_loss` pins essentially exactly at
`5.0000` for the rest of training (all 11 remaining epoch-end readings agree to 4 decimal
places), with only a very slight downward drift visible in the last 2-3 epochs
(4.99981 → 4.99932).

**train_real_score_std** (from the `step`-tagged loop, every 25 batches): starts at 0.0019
(step 0), rises through epoch 1 into a large excursion around the epoch1→2 boundary
(peaking at 0.124 around step 744, coincident with the critic_loss spike noted above),
then drops back to 0.0027 (step 795) and continues declining through epoch 3-4 (0.0021,
0.0009, 0.0011) before **collapsing to the 1e-5 to 2e-5 order of magnitude by step ~1820
(mid-epoch 4) and staying there for the remainder of training** — the last sampled point
(step 8045, epoch 15) is 5.75e-05, essentially unchanged in order of magnitude from
epoch 4 onward despite ~6200 further training steps.

**holdout_real_score_std** (`baseline`/`mid`/`end` tags, one point per eval): 0.00158
(baseline) → 0.0176 (ep1 mid) → 0.026 (ep1 end) → 0.0030 (ep2 mid) → 0.0051 (ep2 end) →
0.0018 (ep3 mid) → 0.0016 (ep3 end) → **1.24e-05 (ep4 mid) → 7.68e-06 (ep4 end)** — collapses
to the same ~1e-5 order as `train_real_score_std` at essentially the same point in
training (epoch 4) — and stays there (order 1e-5 to 6e-5) through epoch 15
(final value 4.85e-05, with a slight uptick visible in the last 1-2 epochs, mirroring the
slight critic_loss drift noted above).

**gen_grad_norm** (per-step, ~3930 logged points): exactly 0 for the entirety of epoch 1
(consistent with the first-epoch-frozen design — both ProtBERT copies are frozen, only the
critic head trains), jumps sharply at step 548 (epoch 2 start) to 4.7, spikes as high as
63.6 at step 566, then settles into a noisy but non-degenerate range roughly 0.3-1.5 for
essentially the whole remainder of training, with occasional larger spikes recurring even
very late (e.g. 2.2 at step 6883, 2.05 at step 7195, 1.57 at step 7670). No trend toward
zero — `gen_grad_norm` stays "alive" (a function of its input) through the entire run,
including all the epochs where `critic_loss`/`*_score_std` are saturated/collapsed.

**kl_loss**: near-zero during the frozen epoch 1 (0.003-0.013), then rises once the
generator starts training (step ~548) into a noisy band roughly 0.02-0.3 for the rest of
training, with intermittent spikes (0.474 at step 2754, 0.348 at step 5937, 0.481 at
step 7539). No sustained drift up or down — stays bounded in the 0-0.5 range described as
healthy in the original lambda_kl=0.05 sweep finding (CLAUDE.md item 18).

**Quality metrics** (structural, `end`-tag only, one point per epoch) stay flat/noisy
across all 15 epochs with no visible degradation trend, despite the critic saturating and
`*_score_std` collapsing from epoch 4/5 onward: plddt_score ranges 0.735-0.776 (final
0.766), scAccuracy 0.382-0.418 (final 0.399), progres 0.888-0.919 (final 0.896),
pairwise_tm 0.720-0.830 (final 0.809), unique_ratio pinned at 1.0 throughout (perfect
sequence uniqueness, every epoch).

**generator_loss** (epoch-end tags): -0.015 (baseline) → -0.075 (ep1) → -0.146 (ep2) →
-0.145 (ep3) → -0.081 (ep4) → then plateaus in a narrow -0.065 to -0.068 band from epoch 5
through epoch 15 (final -0.0625).

## Verdict

`critic_loss` does saturate at (effectively) exactly `5.0000` under this config, holding
there from epoch 5 through the end of the 15-epoch run, consistent with prior sweep
observations. The newly-added held-out validation tooling shows this saturation is
accompanied by a collapse of both `train_real_score_std` and `holdout_real_score_std` from
~0.01-0.03 down to the 1e-5 order of magnitude, with the collapse in both series occurring
at essentially the same point in training (epoch 4, slightly ahead of `critic_loss`'s full
pin at epoch 5) — this is the degenerate-collapse signature flagged in TODO item 20 (the
critic producing near-constant scores regardless of input, rather than a genuine
convergence with healthy score variance). Notably, this collapse happens on both the
training-pool scores and the held-out real sequences the critic never trained on, so it
is not simply memorization of the training set — the critic appears to be collapsing
toward a near-constant output on real sequences generally. `gen_grad_norm` and `kl_loss`
both stay non-degenerate (bounded, non-zero, still noisy) for the entire post-freeze
duration of the run, including all the epochs where the critic-side metrics are
saturated/collapsed — the generator continues receiving a live adversarial gradient signal
even while the critic's discriminative signal (as measured by its output variance) appears
degenerate. Structural quality metrics (plddt, scAccuracy, progres, pairwise_tm,
unique_ratio) stay flat and noisy across the whole run with no visible quality
degradation, and `unique_ratio` is a perfect 1.0 at every logged epoch, so there is no
mode-collapse signal in the generator's output diversity. One clear anomaly: `critic_loss`
spiked to 165.7 at the epoch-2-end checkpoint (step 1076), coincident with a large
transient excursion in `train_real_score_std` (peaking at 0.124 around step 744) — both
this spike and the excursion resolve by epoch 3 without any apparent lasting effect on the
generator-side metrics, but it is a notable instability during the epoch-1-to-2 transition
worth keeping in mind. This run did not have `--holdout_eval_fakes` set, so no
`holdout_fake_score_*` data is available to check real/fake symmetry on the held-out set.
