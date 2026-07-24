# lambda_kl Sweep (2026-07)

**Started:** 2026-07-24
**Status:** Design approved, Phase 1 not yet submitted

## Design

### Goal & Scope

Find and validate a lambda_kl value for the KL anchor (`compute_kl_anchor`,
`models.py`) ahead of the next full-scale training run, now that the generator
gradient-flow fix (`docs/GENERATOR_GRADIENT_FIX.md`) is confirmed working.

- **Seeded mode only** (`10p_train.py`). Blind mode (TODO 16 in `CLAUDE.md`) is
  out of scope for this sweep — validated separately on Anzu first.
- **n_critic fixed at 4** throughout, not swept. Isolates lambda_kl's effect;
  n_critic is deferred to its own separate future sweep to avoid confounding
  two hyperparameters in one experiment.
- Runs on MN5 (H100s). MN5 jobs are hard-capped at 3 days wall clock.

### Prerequisite: pin temperature (implements CLAUDE.md TODO 14)

`models.py:132` used to sample `temperature = min_temp + rand() * (max_temp -
min_temp)` fresh every step (range `[0.8, 1.2]`), adding per-step noise to
`gen_grad_norm`/`kl_loss` that would confound cross-lambda_kl comparison.

- Added a `--temperature` CLI flag to `10p_train.py` and `fully_masked_train.py`,
  default `1.0` (the neutral no-rescale value, and the midpoint of the old
  random range).
- `min_temp`/`max_temp` were hardcoded local variables, never exposed as CLI
  args, so no backward-compat shim was needed — replaced cleanly. The random
  draw is gone from `compute_soft_embeds` (`models.py`), `generate_fakes_for_batch`
  (`10p_train.py`), and `generate_fake_batch` (`fully_masked_train.py`); all
  three now take a single fixed `temperature` parameter.
- Randomized-range sampling was considered and rejected outright (not just
  deferred): existing diversity already comes from `torch.multinomial`
  sampling and per-step random remask-position selection in
  `compute_soft_embeds`; both reference runs (`xwres3cz`/`ydsjq9f4`) held
  `unique_ratio=1.0` under the old random temperature, so it wasn't buying
  diversity the other sources don't already provide. Randomizing a quantity
  that also feeds a differentiable KL loss just adds measurement noise.
- Sweep runs pass `--temperature 1.0` explicitly (also the new default).

### Fixed hyperparameters (all runs, both phases)

| Param | Value | Source |
|---|---|---|
| `n_critic` | 4 | matches reference runs `xwres3cz`/`ydsjq9f4`, isolates lambda_kl |
| `lr_gen` | 5e-6 | conservative pair; generator now receives real adversarial gradient |
| `lr_critic` | 5e-5 | conservative pair |
| `temperature` | 1.0 | fixed flag, see above |
| `max_train_seqs` | None (full dataset, ~52,637 seqs) | time budget confirmed safe, see below |
| `num_eval_sequences` | 30 | bumped from the script default (10) — selection criteria compares per-epoch trends across 5 candidates, and 10 samples is too noisy a signal for that |
| `lambda_gp`, `wd_gen`, `wd_critic`, `batch_size`, `iteration_fill_rate` | script defaults (5.0, 0.01, 0.01, 8, 0.1) | unchanged, not part of this experiment |

### Phase 1 — Screening

5 runs, one per `lambda_kl ∈ {0, 0.005, 0.01, 0.05, 0.1}`:
- `n_epochs=5`, full dataset, all fixed params above.
- SLURM array job (`--array=1-5`), following the existing `n_critic_10p_run.sh`
  pattern — array index maps to a lambda_kl value.
- Script: `slurms/sweeps/lambda_kl_sweep_p1.sh`.
- wandb run names: `kl-sweep-p1-kl0`, `kl-sweep-p1-kl0.005`, `kl-sweep-p1-kl0.01`,
  `kl-sweep-p1-kl0.05`, `kl-sweep-p1-kl0.1`.

### Phase 2 — Confirmation

Top-2 lambda_kl values from Phase 1 (selection criteria below), re-run with
`n_epochs=15`, same fixed params, full dataset.
- SLURM array job (`--array=1-2`).
- Script: `slurms/sweeps/lambda_kl_sweep_p2.sh` (scaffolded with placeholder
  `kl_list=(FILL_ME_1 FILL_ME_2)` — replace with the top-2 lambda_kl values
  once Phase 1's "Winner selection reasoning" below is filled in; do not
  submit as-is).
- wandb run names: `kl-sweep-p2-kl<value>` for each of the 2 winners.

### Selection criteria

A Phase 1 candidate is **disqualified** if:
- `kl_loss` oscillates wildly (unbounded growth or erratic swings, as seen
  pre-fix at 89–1236), or
- `unique_ratio` drops below 1.0 (mode collapse).

Among survivors:
1. **Inspect the per-epoch trend** (all 5 epochs, not just the final one) of
   `plddt`, `scAccuracy`, `progres`, `pairwise_tm`. Deprioritize any candidate
   declining epoch-over-epoch even if its epoch-5 value is numerically ahead of
   a stable/improving candidate — a candidate merely slow to collapse is not
   the same as a stable one (this is exactly how `lambda_kl=0` looked fine
   short-term before quality craters — see `GENERATOR_GRADIENT_FIX.md` Stage 3).
2. Rank remaining candidates by the mean of normalized final-epoch `plddt`,
   `scAccuracy`, `progres`, `pairwise_tm`.
3. Take the top 2 into Phase 2.

### Time budget sanity check

Reference runs `xwres3cz`/`ydsjq9f4`: `max_train_seqs=2000`, 3 epochs,
n_critic=4, ~25 min on an Anzu A6000. Full dataset is ~13x more data than 2000
(seeded mode splits ~52,637 total roughly in half between gen/critic).
Scaling linearly and assuming **no** H100 speedup over the A6000 (pessimistic):

- Phase 1 (5 epochs, full data): ~13x data × 5/3 epochs ≈ 5-6 hours per run.
- Phase 2 (15 epochs, full data): ~13x data × 5 ≈ 27-30 hours per run.

Both comfortably under the 3-day (72h) MN5 limit even under this pessimistic
assumption; H100s are expected to be meaningfully faster than the A6000 for
this workload, adding further margin.

### Open items not covered by this sweep

- n_critic sweep — deferred, separate experiment.
- Blind mode validation (TODO 16) — separate, tested on Anzu first.
- Confidence-inflation reward-hacking risk (`GENERATOR_GRADIENT_FIX.md`, "Open
  risks / caveats") — not diagnosed by this sweep; the proposed max-probability
  logging diagnostic there is a separate follow-up.

---

## Results

_Pending — Phase 1 not yet submitted to MN5._

### Runs

| Run | lambda_kl | Key metric(s) | Verdict | File |
|---|---|---|---|---|
| _(pending)_ | | | | |

### Winner selection reasoning

_Pending._

### Conclusion

_Pending._
