# kl-sweep-p1-kl0.005 (6of3ccvx)

**Date:** 2026-07-25
**Script:** 10p_train.py
**wandb:** https://wandb.ai/ufuktanriverdi1-hacettepe-university/protgen-gan/runs/6of3ccvx

## Config

No hyperparameters were logged to this run's wandb `config` (only wandb/framework
metadata was present — `cli_version 0.28.0`, `python_version 3.12.13`,
`huggingface_version 4.46.3`). Values below are taken from the sweep design doc
(`docs/sweeps/lambda-kl-sweep-2026-07.md`), which fixes these for all Phase 1 arms,
combined with the `lambda_kl` value encoded in the run name.

| Param | Value |
|---|---|
| n_critic | 4 |
| lr_gen | 5e-6 |
| lr_critic | 5e-5 |
| lambda_kl | 0.005 |
| temperature | 1.0 |
| n_epochs | 5 |
| max_train_seqs | None (full dataset, ~52,637 seqs) |

Other fixed-but-unswept params per the design doc: `num_eval_sequences=30`,
`lambda_gp=5.0`, `wd_gen=0.01`, `wd_critic=0.01`, `batch_size=8`,
`iteration_fill_rate=0.1`.

Run finished in ~9,888s (~2h45m) over 2,635 total logged steps.

## Final-epoch metrics

(epoch 5, `_step`=2635, tag="end")

| Metric | Value |
|---|---|
| kl_loss | 0.1783 |
| gen_grad_norm | 0.1062 |
| unique_ratio | 1.0 |
| plddt_score | 0.7690 |
| scAccuracy | 0.4062 |
| progres | 0.9137 |
| pairwise_tm | 0.7996 |
| generator_loss | -0.0706 |
| critic_loss | 4.9979 |

## Per-epoch trend

(values at each epoch's end-of-epoch eval step: 527, 1054, 1581, 2108, 2635)

- plddt_score: 0.748 → 0.743 → 0.763 → 0.760 → 0.769 (epochs 1-5)
- scAccuracy: 0.396 → 0.405 → 0.398 → 0.403 → 0.406 (epochs 1-5)
- progres: 0.911 → 0.914 → 0.911 → 0.910 → 0.914 (epochs 1-5)
- pairwise_tm: 0.791 → 0.812 → 0.802 → 0.812 → 0.800 (epochs 1-5)
- unique_ratio: 1.0 → 1.0 → 1.0 → 1.0 → 1.0 (epochs 1-5, no mode collapse at any point)
- generator_loss: -0.111 → -0.115 → -0.170 → -0.116 → -0.071 (epochs 1-5)
- critic_loss: 5.047 → 5.111 → 4.982 → 5.007 → 4.998 (epochs 1-5)
- kl_loss (raw per-step samples, not epoch-aligned): epoch-1 window (steps 13-524)
  ranged ~0.002-0.007; epoch-5 window (steps 2205-2623) ranged ~0.14-0.32. Grew
  roughly two orders of magnitude over training but stayed bounded — no wild
  oscillation or unbounded growth of the kind seen pre-fix (89-1236).
- gen_grad_norm (raw per-step samples, not epoch-aligned): exactly 0 throughout
  the epoch-1 window (steps 40-484), consistent with the first-epoch freezing
  design. Nonzero from epoch 2 onward; epoch-5 window (steps 2126-2619) ranged
  ~0.07-1.94.

## Verdict

This run (lambda_kl=0.005) completed all 5 epochs without triggering either Phase-1
disqualification criterion: `unique_ratio` held at 1.0 throughout (no mode collapse)
and `kl_loss`, while it grew roughly two orders of magnitude from epoch 1 (~0.002-0.007)
to epoch 5 (~0.14-0.32), stayed bounded rather than oscillating wildly or diverging.
`gen_grad_norm` was 0 during the frozen first epoch as designed, then became
consistently nonzero from epoch 2 onward, confirming the generator received
adversarial gradient. Structural quality metrics (plddt_score, scAccuracy, progres,
pairwise_tm) were all flat-to-slightly-improving across the 5 epochs rather than
declining — plddt_score rose from 0.748 to 0.769, scAccuracy from 0.396 to 0.406,
progres stayed essentially constant around 0.91, and pairwise_tm fluctuated in a
narrow 0.79-0.81 band with no downward trend. No NaNs or anomalies were observed in
the sampled history. One caveat for downstream analysis: this run's wandb `config`
block did not capture the actual CLI hyperparameters (only wandb/framework
metadata was logged), so the config values reported here are inferred from the
sweep design doc and the run name rather than read directly from wandb config.
