# kl-sweep-p2-retry-kl0.05 (2rohdbx2)

**Date:** 2026-07-31
**Script:** 10p_train.py (seeded mode)
**wandb:** https://wandb.ai/ufuktanriverdi1-hacettepe-university/protgen-gan/runs/2rohdbx2

## Sweep context

Retry of the `lambda_kl=0.05` arm of Phase 2 of the lambda_kl sweep
(`docs/sweeps/lambda-kl-sweep-2026-07.md`), tagged `kl-sweep-p2` and
`kl-sweep-p2-retry` in wandb. The original Phase 2 run for this arm
(`kl-sweep-p2-kl0.05`, `228dww56`, documented in
`docs/runs/kl-sweep-p2-kl0.05_228dww56.md`) had two problems, both fixed before
this retry:

1. **Temperature-pinning bug:** `generate_fake_sequences` in `val_metrics.py`
   (used by `run_evaluation()` to generate the sequences behind
   `plddt`/`scAccuracy`/`progres`/`pairwise_tm`/`unique_ratio`) ignored the
   `--temperature` CLI flag and drew a fresh random `uniform(0.8, 1.2)`
   temperature at every fill step instead. So every quality metric from the
   entire original sweep (both phases) was measured under uncontrolled
   temperature. Fixed 2026-07-30. `gen_grad_norm`/`kl_loss`/`critic_loss`/
   `generator_loss` were unaffected by this bug (they use the pinned training-loop
   generation path, not `generate_fake_sequences`).
2. **Disk-quota checkpoint loss:** both original Phase 2 runs hit an MN5
   `gpfs_projects` disk-quota error while saving the epoch-15 checkpoint to disk.
   Training/eval/wandb-logging completed fine for both; only the on-disk model
   weights were lost. This retry's checkpoint **did** save successfully (confirmed
   via SLURM logs) — no quota issue this time.

This run is the `lambda_kl=0.05` arm of the retry, submitted via
`slurms/sweeps/lambda_kl_sweep_p2.sh` (array task 2).

## Config

| Param | Value |
|---|---|
| n_critic | 4 |
| lr_gen | 5e-6 |
| lr_critic | 5e-5 |
| lambda_kl | 0.05 |
| temperature | 1.0 |
| n_epochs | 15 |
| max_train_seqs | None (full dataset, ~52,637 seqs) |
| num_eval_sequences | 30 |
| batch_size, lambda_gp, wd_gen, wd_critic, iteration_fill_rate | script defaults (8, 5.0, 0.01, 0.01, 0.1) — unchanged, not part of this sweep |

**Config provenance caveat:** `run.config` (dict) is empty for this run, and
`run._attrs['rawconfig']` also contains nothing but `_wandb` client/framework
metadata (`cli_version`, `python_version`, `huggingface_version`, etc.) — no
run-specific keys like `lambda_kl`, `lr_gen`, etc. This was checked directly via
the wandb Python API (`api.run(...)`), **not** the MCP server, which was
unreachable this session (DNS resolution failure for `mcp.withwandb.com`). This
means the empty-config gap is **not** an MCP-tool-only limitation, contrary to
the "corrected" explanation in `docs/sweeps/lambda-kl-sweep-2026-07.md`
(2026-07-25 note) and repeated in the original `kl-sweep-p2-kl0.05` run doc —
that explanation was not independently re-verified here and should not be taken
as settled. The values in the table above are backfilled with certainty from
`slurms/sweeps/lambda_kl_sweep_p2.sh` (the actual submitted job script for this
run, array task 2, `lambda_kl=0.05`), not from wandb's `config` field — treat as
unverified-by-wandb-API but not fabricated.

## Final-epoch metrics (epoch 15, step 7905)

| Metric | Value |
|---|---|
| kl_loss | 0.1034 |
| gen_grad_norm | 1.2694 |
| unique_ratio | 1.0 |
| plddt_score | 0.7609 |
| scAccuracy | 0.4046 |
| progres | 0.9143 |
| pairwise_tm | 0.8290 |
| critic_loss | 4.9999 |
| generator_loss | 0.7286 |

## Per-epoch trend

(baseline = pre-epoch-1 eval, `tag="baseline"`, step 2; remaining points are
epoch-end evals, `tag="end"`, one per epoch, epochs 1-15)

- plddt_score: 0.754 (baseline) → 0.734, 0.752, 0.725, 0.750, 0.787, 0.719, 0.741, 0.750, 0.744, 0.760, 0.754, 0.753, 0.725, 0.741, 0.761 (epochs 1-15) — flat/noisy around 0.72-0.79, no directional trend, ends slightly above baseline.
- scAccuracy: 0.395 (baseline) → 0.382, 0.400, 0.383, 0.402, 0.401, 0.380, 0.388, 0.401, 0.396, 0.409, 0.397, 0.399, 0.372, 0.392, 0.405 (epochs 1-15) — flat, oscillating narrowly around 0.37-0.41.
- progres: 0.917 (baseline) → 0.913, 0.917, 0.909, 0.916, 0.917, 0.906, 0.909, 0.894, 0.892, 0.902, 0.918, 0.913, 0.890, 0.914, 0.914 (epochs 1-15) — flat/stable around 0.89-0.92, mild dip epochs 8-9 and 13, recovers thereafter.
- pairwise_tm: 0.805 (baseline) → 0.729, 0.795, 0.773, 0.791, 0.782, 0.740, 0.785, 0.808, 0.767, 0.837, 0.781, 0.806, 0.673, 0.785, 0.829 (epochs 1-15) — noisy, no clear decline; run-low 0.673 at epoch 13, recovers to a near-run-high 0.829 at epoch 15 (final).
- unique_ratio: 1.0 at every logged epoch (baseline through epoch 15) — flat, no mode collapse.
- critic_loss (epoch-end eval value): 4.971 (baseline) → 8.230, 4.988, 4.999, 5.000, 5.000, 5.000, 4.999, 5.000, 4.999, 5.000, 5.000, 5.000, 4.999, 5.000, 4.999 (epochs 1-15) — one transient spike at epoch 1 (8.23), then settles to ~4.988-5.000 from epoch 2 onward and stays there for the rest of training. This is earlier and more persistent saturation than the original run (which didn't fully pin until epoch 9).
- generator_loss: 0.029 (baseline) → 0.025, 0.109, -0.011, -0.025, -0.022, 0.033, 0.141, 0.199, 0.491, 1.146, 1.600, 1.407, 0.784, 2.289, 0.729 (epochs 1-15) — noisy, trends upward/more-positive from epoch 9 onward (unlike the original run, whose generator_loss went increasingly negative over the same window); ends epoch 15 well above baseline.
- kl_loss: logged per-batch (263 points/epoch). Per-epoch range: epoch 1 ~0.002-0.034 (mean 0.005, low — first-epoch freezing), epoch 2 ~0.003-0.356 (mean 0.053), then a gradual rise settling into roughly 0.02-0.5 per-epoch range from epoch 4 onward, with mean climbing slowly from ~0.09 (epoch 4) to ~0.12 (epoch 15). One late spike to 0.62 at epoch 15 (single batch), otherwise consistent with the original run's bounded 0.03-0.20-ish range — no unbounded growth, no return of the pre-fix 89-1236 oscillation.
- gen_grad_norm: logged per-batch (263 points/epoch). Epoch 1 is exactly 0 throughout (first-epoch freezing — expected, matches training design). From epoch 2 onward, per-epoch mean is stable in the 0.74-1.01 range for every single epoch through epoch 15 (epoch 15 mean 1.01, min 0.40, max 14.26 — one isolated spike, similar in character to the original run's isolated 15.0 spike at epoch 7). Overall run min/max/mean (excluding the frozen epoch-1 zeros): min ~0.17, max ~14.26 (one outlier), mean ~0.87. No collapse toward zero at any point after epoch 1.

## Verdict

This retry reproduces the original run's core finding, and in one respect makes
the case for `lambda_kl=0.05` *stronger*, not weaker, now that the temperature
bug is fixed. Critic saturation (`critic_loss` pinned at ~4.99-5.00) sets in
**earlier and more persistently** here than in the original run — by epoch 2
instead of epoch 9, and essentially every single epoch from 2 through 15 sits at
or above 4.988, versus the original's gradual saturation with genuine variation
(4.91-4.99) through epoch 8. Despite this near-total saturation across almost
the entire training run, `gen_grad_norm` never collapsed: its per-epoch mean
stayed in a tight 0.74-1.01 band from epoch 2 through epoch 15, essentially
indistinguishable from the original run's reported 0.4-2.0 range, with isolated
high spikes (14.26 here vs. 15.0 in the original) that did not destabilize
subsequent epochs either time. This is the key mechanistic result: the
generator keeps receiving meaningful, non-vanishing adversarial gradient even
when the critic is saturated almost the whole run, not just in the back half —
the strongest version yet of the claim that justified picking `lambda_kl=0.05`
over `lambda_kl=0`/`0.005` in the first place.

The quality metrics (plddt_score, scAccuracy, progres, pairwise_tm) tell
essentially the same story as the original run: flat and noisy across all 15
epochs, no sustained decline, ending near or slightly above their epoch-1/
baseline values. Absolute levels are close to the original run's
(plddt 0.761 vs 0.767, scAccuracy 0.405 vs 0.399, progres 0.914 vs 0.911,
pairwise_tm 0.829 vs 0.837, all final-epoch) — the temperature fix did not
substantially shift these metrics' magnitude or trend shape, which is reassuring
given that both runs' *quality* numbers were previously suspect due to the
random-temperature confound. `unique_ratio` held at 1.0 throughout, matching
the original — no mode collapse. `kl_loss` stayed bounded (no unbounded growth,
no return of the pre-fix oscillation), consistent with the KL anchor fix
functioning correctly. One notable difference: `generator_loss` trended
increasingly *positive* from epoch 9 onward here, versus increasingly *negative*
in the original run over the same window — worth flagging as a sign that the
generator/critic dynamic isn't perfectly reproducible run-to-run even under
identical hyperparameters and seed-free config, though it did not translate into
a difference in the quality metrics or in `gen_grad_norm` health.

Overall: this retry confirms `lambda_kl=0.05` is the right pick going into
full-scale training — `gen_grad_norm` remains healthy even under critic
saturation that is now more severe and more persistent than previously observed,
and this run's checkpoint (unlike the original's) actually saved to disk, so it
is usable for downstream generation if needed. The critic-saturation-at-5.000
phenomenon itself (item 20 in `CLAUDE.md`'s TODO) remains unexplained and, if
anything, this retry's earlier onset makes it a slightly higher-priority
follow-up than before.
