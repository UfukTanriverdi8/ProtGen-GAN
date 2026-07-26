# kl-sweep-p2-kl0.005 (s9n1tlld)

**Date:** 2026-07-26 (synced; trained ~2026-07-26 per wandb createdAt)
**Script:** 10p_train.py
**wandb:** https://wandb.ai/ufuktanriverdi1-hacettepe-university/protgen-gan/runs/s9n1tlld

## Config

wandb's config API (both `get_run_history_tool` and a direct GraphQL `query_wandb_tool`
query) returned only `_wandb` client/framework metadata (`cli_version`, `python_version`,
`huggingface_version`, etc.) for this run — no run-specific hyperparameter keys were
retrievable from either tool. The values below are taken from the Phase 2 sweep design
(`docs/sweeps/lambda-kl-sweep-2026-07.md`), which specifies fixed parameters for all Phase 2
runs, confirmed against this run's name (`kl-sweep-p2-kl0.005`) and its `kl-sweep-p2` tag.

| Param | Value |
|---|---|
| n_critic | 4 |
| lr_gen | 5e-6 |
| lr_critic | 5e-5 |
| lambda_kl | 0.005 |
| temperature | 1.0 |
| n_epochs | 15 |
| num_eval_sequences | 30 |
| max_train_seqs | full dataset (not capped) |
| mode | seeded (10p_train.py) |

## Final-epoch metrics

Final logged values, epoch 15 (`_step` 7905, summary tag `"end"`):

| Metric | Value |
|---|---|
| kl_loss | 0.1317 |
| gen_grad_norm | 0.0988 |
| unique_ratio | 1.0 |
| plddt_score | 0.7470 |
| scAccuracy | 0.3855 |
| progres | 0.9114 |
| pairwise_tm | 0.7662 |
| critic_loss | 5.0000 |
| generator_loss | -0.2093 |

## Per-epoch trend

Eval metrics are logged once per epoch (epochs 1-15, all 15 present and confirmed in
wandb history — see Verdict below on the disk-quota crash).

- plddt_score: 0.747 → 0.746 → 0.750 → 0.729 → 0.755 → 0.755 → 0.758 → 0.739 → 0.751 → 0.742 → 0.740 → 0.756 → 0.733 → 0.732 → 0.746 (epochs 1-15). Flat/oscillating in a narrow 0.729-0.758 band, no directional decline.
- scAccuracy: 0.387 → 0.383 → 0.401 → 0.376 → 0.401 → 0.398 → 0.403 → 0.393 → 0.394 → 0.401 → 0.384 → 0.377 → 0.377 → 0.391 → 0.386 (epochs 1-15). Similarly flat, ~0.38-0.40 band throughout.
- progres: 0.911 → 0.906 → 0.911 → 0.909 → 0.917 → 0.917 → 0.918 → 0.888 → 0.916 → 0.920 → 0.910 → 0.900 → 0.899 → 0.906 → 0.911 (epochs 1-15). Stable ~0.89-0.92, single dip at epoch 7 (0.888), recovers immediately.
- pairwise_tm: 0.782 → 0.754 → 0.739 → 0.704 → 0.774 → 0.798 → 0.777 → 0.770 → 0.773 → 0.793 → 0.794 → 0.776 → 0.746 → 0.765 → 0.766 (epochs 1-15). Dips to a low of 0.704 at epoch 3, recovers and oscillates 0.75-0.80 for the remainder — no sustained decline.
- unique_ratio: 1.0 at every epoch (1-15). No mode collapse across the full 15-epoch run.
- kl_loss (per-batch, not per-epoch-aligned in the logged history): starts low (~0.004-0.03) through roughly the first third of training, then rises and oscillates in a wider 0.02-0.65 band from roughly step 900 onward (coinciding with `gen_grad_norm` becoming consistently nonzero), settling around 0.04-0.15 for most of the back half with occasional spikes (e.g. 0.655 at step 3259, 0.374 at step 4721). Ends at 0.132 (epoch 15 summary).
- gen_grad_norm (per-batch): 0 for early steps (generator not yet receiving nonzero adversarial gradient, consistent with first-epoch freezing behavior), turns nonzero from step ~637 onward and stays in a 0.03-0.25 range for most of training with a few outlier spikes (11.9 at step 1005, 3.35 at step 1100) early in that transition, then settles to a steadier ~0.05-0.12 band by the second half. Ends at 0.099 (epoch 15 summary).
- critic_loss: 4.95 (epoch 1) → 16.35 (epoch 1, later batch) → 4.93 → 4.996 → 5.66 → then pins at 5.0000 (to 6+ decimal places) for every epoch from epoch 5 through epoch 15.
- generator_loss: -0.016 → -0.012 → -0.060 → -0.116 → -0.139 → -0.215 → -0.215 → -0.214 → -0.214 → -0.213 → -0.212 → -0.212 → -0.211 → -0.211 → -0.209 (epochs 1-15). Settles to a very stable ~-0.21 plateau from epoch 5 onward.

## Verdict

All 15 epochs of metrics are confirmed present in this run's wandb history (`unique_ratio`,
`plddt_score`, `scAccuracy`, `progres`, `pairwise_tm`, `critic_loss`, `generator_loss` all
have an epoch-15 data point, and the run summary carries `epoch: 15` with `tag: "end"`),
confirming that the disk-quota-exceeded error hit only the final on-disk checkpoint save
(`safetensors_rust.SafetensorError`) after training, evaluation, and wandb logging for
epoch 15 had already completed — the epoch-15 model weights themselves were never written
to disk and are not available for future generation runs from this checkpoint, but no
training or evaluation data was lost. Quality metrics (plddt_score, scAccuracy, progres,
pairwise_tm) all stayed roughly flat/oscillating across the full 15 epochs with no sustained
upward or downward trend, and unique_ratio held at 1.0 throughout — no mode collapse over
the longer horizon. gen_grad_norm turned nonzero partway through epoch ~2-3 (consistent
with first-epoch freezing) and stayed bounded thereafter aside from a few early transient
spikes. One clear anomaly: critic_loss pinned exactly at 5.0000 (to several decimal places)
for every epoch from epoch 5 through epoch 15, alongside generator_loss settling to a very
stable ~-0.21 plateau over the same span — both consistent with the critic saturating and
the WGAN critic/generator loss dynamic effectively flatlining for the back two-thirds of
training, similar to the saturation pattern noted for lambda_kl=0 in Phase 1, but occurring
here despite lambda_kl=0.005 being nonzero. kl_loss itself grew noisier and shifted to a
higher band (roughly 0.02-0.65, occasional spikes) after the first ~900 steps compared to
its low, tight range early in training, though it did not diverge or trend monotonically
upward through epoch 15.
