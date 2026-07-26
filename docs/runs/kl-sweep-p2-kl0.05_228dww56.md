# kl-sweep-p2-kl0.05 (228dww56)

**Date:** 2026-07-26
**Script:** 10p_train.py (seeded mode)
**wandb:** https://wandb.ai/ufuktanriverdi1-hacettepe-university/protgen-gan/runs/228dww56

## Sweep context

Phase 2 of the lambda_kl sweep (`docs/sweeps/lambda-kl-sweep-2026-07.md`). Phase 1
screened `lambda_kl ∈ {0, 0.005, 0.01, 0.05, 0.1}` over 5 epochs and selected
`{0.005, 0.05}` as the top-2 for a 15-epoch confirmation run. This run is the
`lambda_kl=0.05` arm of that confirmation phase, tagged `kl-sweep-p2` in wandb.

**Operational note:** this run hit a `safetensors_rust.SafetensorError: I/O error:
Disk quota exceeded` on MN5 during the epoch-15 **checkpoint save to disk**, at the
very end of the job. Training and evaluation for all 15 epochs completed
successfully before that point — wandb logging happens before the checkpoint-save
step in the training loop. Confirmed below: epoch 15's full metric set (`kl_loss`,
`gen_grad_norm`, `unique_ratio`, `plddt_score`, `scAccuracy`, `progres`,
`pairwise_tm`, `critic_loss`, `generator_loss`) is present in this run's history at
step 7905 (`epoch=15`, `tag="end"`), matching the run's summary metrics exactly, and
wandb's run `state` is `finished`. Only the final on-disk model checkpoint (weights)
was lost — this checkpoint is not available for future generation runs.

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

**Config provenance caveat:** wandb's `config` field for this run returns only
`_wandb` client/framework metadata (`cli_version`, `python_version`, etc.) via both
`get_run_history_tool` and a direct `query_wandb_tool` GraphQL query — no
run-specific keys (`lambda_kl`, `lr_gen`, etc.) are surfaced through either tool.
This matches the known gap already documented and corrected-for in
`docs/sweeps/lambda-kl-sweep-2026-07.md` (Results section, 2026-07-25 correction):
the actual hyperparameters were confirmed present and correct in the raw synced
`offline-run-*/files/config.yaml` on Anzu — nothing was lost in offline logging or
sync, only in how these MCP tools surface the `config` field for this project. The
values above are backfilled from the run name (`kl-sweep-p2-kl0.05`) and the sweep
design doc's fixed-parameter table, per the same convention used for every Phase 1
run doc — treat as unverified-by-wandb-API but not fabricated.

## Final-epoch metrics (epoch 15, step 7905)

| Metric | Value |
|---|---|
| kl_loss | 0.1445 |
| gen_grad_norm | 1.1306 |
| unique_ratio | 1.0 |
| plddt_score | 0.7671 |
| scAccuracy | 0.3987 |
| progres | 0.9105 |
| pairwise_tm | 0.8372 |
| critic_loss | 5.0000 |
| generator_loss | -1.2155 |

## Per-epoch trend

(baseline = pre-epoch-1 eval, `tag="baseline"`, step 2; remaining points are
epoch-end evals, `tag="end"`, one per epoch, epochs 1-15)

- plddt_score: 0.757 (baseline) → 0.736, 0.772, 0.752, 0.743, 0.759, 0.754, 0.755, 0.751, 0.754, 0.763, 0.751, 0.738, 0.754, 0.747, 0.767 (epochs 1-15) — flat/noisy around 0.74-0.77, no directional trend, ends near baseline.
- scAccuracy: 0.393 (baseline) → 0.383, 0.404, 0.390, 0.406, 0.392, 0.393, 0.402, 0.391, 0.394, 0.408, 0.413, 0.401, 0.406, 0.387, 0.399 (epochs 1-15) — flat, oscillating narrowly around 0.39-0.41.
- progres: 0.906 (baseline) → 0.906, 0.918, 0.920, 0.904, 0.886, 0.916, 0.897, 0.901, 0.912, 0.915, 0.919, 0.913, 0.915, 0.915, 0.911 (epochs 1-15) — flat/stable around 0.90-0.92, one dip to 0.886 at epoch 5, recovers thereafter.
- pairwise_tm: 0.824 (baseline) → 0.760, 0.810, 0.752, 0.834, 0.745, 0.764, 0.757, 0.761, 0.798, 0.772, 0.802, 0.767, 0.767, 0.725, 0.837 (epochs 1-15) — noisy, no clear decline; run-low 0.725 at epoch 14, but recovers to a run-high 0.837 at epoch 15 (final).
- unique_ratio: 1.0 at every logged epoch (baseline through epoch 15) — flat, no mode collapse.
- critic_loss: 4.952 (baseline) → 7.770, 5.092, 4.938, 4.910, 4.994, 4.999, 4.975, 4.990, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000 (epochs 1-15) — settles to exactly 5.000 from epoch 9 onward (critic saturation).
- generator_loss: 0.120 (baseline) → 0.108, 0.098, 0.077, 0.695, 0.677, 1.283, -0.629, -0.851, -1.077, -1.088, -1.093, -1.092, -1.127, -1.137, -1.216 (epochs 1-15) — rises through epoch 6, then flips negative and steadily decreases (more negative) from epoch 7 through 15.
- kl_loss and gen_grad_norm are logged per-batch (multiple points per epoch), not per-epoch-eval, so are not directly epoch-aligned with the metrics above. Across the full run, `kl_loss` stayed bounded, mostly in the 0.03-0.20 range with occasional spikes up to ~0.27 (early) — no unbounded growth or the pre-fix 89-1236 oscillation pattern. `gen_grad_norm` stayed mostly in the 0.4-2.0 range throughout, with one isolated spike to 15.0 around step 3731 (mid-training, ~epoch 7) that did not recur or destabilize training afterward; in the final epoch (steps ~7378-7905) it ranged 0.46-1.43, i.e. still comfortably nonzero, not collapsed.

## Verdict

This run completed all 15 epochs of training and evaluation successfully (wandb
state `finished`, epoch-15 metrics present and matching the run summary); only the
final checkpoint save to disk was lost to an MN5 disk-quota error at the very end,
so no generation runs can be performed from this specific 15-epoch checkpoint.
`unique_ratio` held at 1.0 throughout — no mode collapse. The four quality metrics
(plddt_score, scAccuracy, progres, pairwise_tm) all stayed essentially flat and
noisy across all 15 epochs, ending epoch 15 near or slightly above their epoch-1
values, with no sustained decline — this is a materially different shape from the
Phase 1 `lambda_kl=0.05` run's "mild late-epoch dip" over 5 epochs. One notable
pattern: `critic_loss` settles to exactly 5.000 from epoch 9 onward, the same
saturation signature Phase 1 flagged as concerning for `lambda_kl=0` — but here
`gen_grad_norm` did not collapse alongside it (it stayed in the 0.4-2.0 range
through the final epoch), so the generator continued receiving nonzero adversarial
gradient despite critic saturation, unlike the disqualified `lambda_kl=0` Phase 1
run. `kl_loss` stayed bounded throughout, consistent with the KL anchor fix.
