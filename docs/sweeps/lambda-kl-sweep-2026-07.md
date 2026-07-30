# lambda_kl Sweep (2026-07)

**Started:** 2026-07-24
**Status:** Phase 1 and Phase 2 complete and analyzed (2026-07-26). **Winner: `lambda_kl=0.05`**
(reverses Phase 1's provisional pick of `0.005` — see Phase 2 Results below). Neither Phase 2
run's final checkpoint was saved to disk (MN5 `gpfs_projects` quota exceeded mid-sync) — a
fresh training run is required before this winner can be used for generation.

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

**Correction (2026-07-30):** the temperature-pinning fix (`CLAUDE.md` TODO 14) that this
sweep depended on to remove a confounding variable was incomplete. It correctly pinned
temperature in the training-loop generation path, but missed `generate_fake_sequences`
(`val_metrics.py`), which produced every sequence behind the quality metrics below
(`plddt`, `scAccuracy`, `progres`, `pairwise_tm`, `unique_ratio`) across both phases of
this sweep — it drew an uncontrolled `uniform(0.8, 1.2)` temperature every fill step
regardless of `--temperature`. `gen_grad_norm`, `kl_loss`, `critic_loss`, and
`generator_loss` were unaffected (correctly pinned throughout), so the mechanistic case
for `lambda_kl=0.05` (its `gen_grad_norm` staying healthy after critic saturation, unlike
`0.005`'s) still holds. But every quality-metric number and comparison below — including
the Phase 1→Phase 2 reversal — was measured under this uncontrolled temperature, adding
an unquantified confound on top of the small-sample noise already flagged for close
calls. Bug fixed 2026-07-30 (`generate_fake_sequences` now takes and uses a `temperature`
parameter). Treat the quality-metric comparisons below as suggestive, not confirmed;
`lambda_kl=0`'s disqualification is likely still correct given the size of its collapse,
but the closer calls among survivors are less certain than presented.

All 5 Phase 1 arms completed 5 epochs on the full dataset with no NaNs and no
hard-criteria disqualification (`unique_ratio` stayed at 1.0 in every run;
`kl_loss`, where computed, stayed bounded — nowhere near the pre-fix 89–1236
oscillation range). Selection came down to the trend-inspection and
final-epoch composite steps.

**Correction (2026-07-25):** the per-run docs below note that wandb `config`
only returned `_wandb` client/framework metadata, with hyperparameters
backfilled from the run name instead. Checked the raw synced run data
directly (`arrivals/offline-run-*/files/config.yaml`) and the actual
hyperparameters (`lambda_kl`, `lr_gen`, `lr_critic`, etc.) were logged
correctly by `10p_train.py`'s `wandb.config.update()` call and are present in
every run's config — nothing was lost in offline logging or sync. The gap
was in the review subagents' wandb MCP query (likely `get_run_history_tool`
surfacing only the `_wandb` block, not the flattened custom keys alongside
it), not in the training pipeline. The backfilled values in the per-run docs
happen to be correct since they were sourced from the run name/design doc,
but should be treated as unverified-by-wandb rather than confirmed. Fixed by
having `wandb-run-reviewer` fall back to `query_wandb_tool` or a direct read
of `files/config.yaml` when `get_run_history_tool`'s config comes back
metadata-only.

### Runs

| Run | lambda_kl | Key metric(s) (final epoch) | Verdict | File |
|---|---|---|---|---|
| `kl-sweep-p1-kl0` | 0 | plddt 0.67 (from 0.76 baseline), pairwise_tm 0.67 (low of 0.61 at ep4) | Deprioritized — quality decline + `gen_grad_norm` collapsed to ~1e-7 for epochs 3-5 while `critic_loss` saturated at exactly 5.000, i.e. effectively no adversarial signal for the back half of training | `docs/runs/kl-sweep-p1-kl0_hobeuiv1.md` |
| `kl-sweep-p1-kl0.005` | 0.005 | plddt 0.77 (rising from 0.75), scAcc 0.41, pairwise_tm 0.80 | **Winner (1st)** — only arm with stable-to-improving trend on all 4 quality metrics simultaneously; also wins or near-ties every final-epoch metric outright | `docs/runs/kl-sweep-p1-kl0.005_6of3ccvx.md` |
| `kl-sweep-p1-kl0.01` | 0.01 | plddt 0.74, pairwise_tm 0.71 (low of run) | Survives hard gate, but noisiest/no clear direction; lowest composite score of the 4 survivors | `docs/runs/kl-sweep-p1-kl0.01_kwm7j52c.md` |
| `kl-sweep-p1-kl0.05` | 0.05 | plddt 0.74, pairwise_tm 0.74 (down from ep4 peak of 0.80) | **Winner (2nd)** — mild late-epoch dip in pairwise_tm, but edges out 0.1 on composite | `docs/runs/kl-sweep-p1-kl0.05_v7ha9aab.md` |
| `kl-sweep-p1-kl0.1` | 0.1 | plddt 0.74, scAcc 0.38 (down from 0.40 peak), pairwise_tm 0.74 | Survives hard gate; similar mild late decline shape to 0.05, narrowly loses on composite | `docs/runs/kl-sweep-p1-kl0.1_0yeu4yiu.md` |

### Winner selection reasoning

Per the Selection criteria above: none of the 5 runs hit either hard
disqualifier, so all 5 went through trend inspection first.

**`lambda_kl=0` deprioritized by trend, not by the hard gate.** `unique_ratio`
never dropped and `kl_loss` isn't computed at `lambda_kl=0`, so it technically
survives the two written disqualifiers — but `plddt_score` dropped sharply
after epoch 1 (0.76→0.67) and never recovered, and `pairwise_tm` declined to
a run-low of 0.61 before a partial rebound. More importantly, this is exactly
the "merely slow to collapse" pattern the design doc's criteria are meant to
catch: `gen_grad_norm` spiked once mid-epoch-2 then collapsed to ~1e-7 for
epochs 3-5, coinciding with `critic_loss` sitting exactly at 5.000 — the
critic saturated and the generator got no meaningful adversarial gradient for
60% of training. This matches CLAUDE.md's prior finding that `lambda_kl=0`
runs decline in quality even with `unique_ratio=1.0` throughout — confirmed
again here, with a specific mechanism attached this time.

**Remaining 4 candidates ranked by mean of normalized final-epoch metrics**
(plddt, scAccuracy, progres, pairwise_tm; min-max normalized across the 4
survivors per metric):

| lambda_kl | Composite score |
|---|---|
| 0.005 | 1.00 |
| 0.05 | 0.26 |
| 0.1 | 0.25 |
| 0.01 | 0.00 |

`0.005` isn't a marginal winner — it has the best or tied-best value on 3 of
4 final-epoch metrics (plddt, scAccuracy, pairwise_tm) and is a close second
on `progres`, and its per-epoch trend is the only one that's genuinely
stable-to-improving across the board rather than flat/noisy or mildly
declining. `0.05` and `0.1` are close (0.26 vs 0.25 — within likely noise
given `num_eval_sequences=30`); `0.05` takes the edge on the mechanical
ranking, but this margin is thin enough not to be a confident call on its
own — Phase 2's longer horizon (15 epochs) is exactly the kind of
confirmation this warrants. `0.01` ranks last among the survivors: noisiest
trajectory, no clear direction, `pairwise_tm` ending at its run-low.

### Conclusion (Phase 1)

**Phase 2 candidates: `lambda_kl ∈ {0.005, 0.05}`.** `0.005` is the clear
front-runner; `0.05` is a reasonable second pick over `0.1` but by a thin
margin worth re-checking at 15 epochs rather than treating as settled.
`lambda_kl=0` is confirmed (again, with a new mechanistic detail — critic
saturation cutting off gradient) as unsuitable without the KL anchor.
`slurms/sweeps/lambda_kl_sweep_p2.sh` updated with `kl_list=(0.005 0.05)`.

---

## Phase 2 Results (2026-07-26)

Both arms ran the full `n_epochs=15` on MN5 and completed all training and evaluation
successfully — confirmed via wandb history (`state=finished`, epoch-15 metrics present and
matching each run's summary). **Operational issue, not a training issue:** both jobs hit
`safetensors_rust.SafetensorError: I/O error: Disk quota exceeded` on MN5's `gpfs_projects`
filesystem (confirmed via `bsc_quota`: usage had reached the 4.20 TB hard limit, driven by
`10p_train.py` saving a full checkpoint — both ProtBERT copies plus both optimizer states,
~8.2–8.5 GB — every epoch with no cleanup) while writing the epoch-15 checkpoint to disk,
*after* that epoch's training/eval/wandb-logging had already completed. Consequence: **all
15 epochs of metrics are valid and complete for both runs, but neither run's final model
weights were ever written to disk** — a fresh run is needed to actually obtain a usable
checkpoint from the winning arm.

### Runs

| Run | lambda_kl | Key metric(s) (epoch 15) | Verdict | File |
|---|---|---|---|---|
| `kl-sweep-p2-kl0.005` | 0.005 | plddt 0.747, scAcc 0.386, progres 0.911, pairwise_tm 0.766, gen_grad_norm 0.099 | Flat/stable on all 4 quality metrics, `unique_ratio=1.0` throughout — but `critic_loss` pins at exactly 5.000 from epoch 5 on, and `gen_grad_norm` settles to a comparatively weak ~0.03–0.25 band for the back two-thirds of training | `docs/runs/kl-sweep-p2-kl0.005_s9n1tlld.md` |
| `kl-sweep-p2-kl0.05` | 0.05 | plddt 0.767, scAcc 0.399, progres 0.911, pairwise_tm 0.837, gen_grad_norm 1.131 | **Winner** — flat/stable on all 4 quality metrics, `unique_ratio=1.0` throughout, ends at or above baseline on every metric; `critic_loss` also pins at 5.000 (from epoch 9 on) but `gen_grad_norm` stays healthy (~0.4–2.0) through the final epoch — the critic saturating did not cut off the generator's adversarial signal here | `docs/runs/kl-sweep-p2-kl0.05_228dww56.md` |

### Winner selection reasoning

Phase 1 picked `0.005` as the clear front-runner on a 5-epoch horizon. At 15 epochs, that
ranking reverses: `0.05` beats `0.005` on 3 of 4 final-epoch quality metrics outright (plddt
0.767 vs 0.747, scAccuracy 0.399 vs 0.386, pairwise_tm 0.837 vs 0.766 — the largest gap of
the four) and ties on the fourth (progres, 0.911 vs 0.911). Both runs are `unique_ratio=1.0`
throughout with no mode collapse, so neither is disqualified by the hard gate.

The more important signal is mechanistic, not just the final numbers: **both runs show the
same critic-saturation pattern** (`critic_loss` pinned exactly at `5.000`) that got
`lambda_kl=0` disqualified in Phase 1 — but the two runs diverge sharply in what that
saturation does to the generator. In `0.005`, `gen_grad_norm` settles to a comparatively weak
~0.03–0.25 once the critic saturates (epoch 5 onward), while in `0.05` it stays in a much
healthier ~0.4–2.0 band all the way through epoch 15 despite the critic saturating even
earlier (epoch 9). In other words: `0.05`'s stronger KL anchor appears to keep the generator
receiving a meaningful adversarial-plus-anchor gradient even once the critic stops
discriminating well, whereas `0.005`'s weaker anchor leaves the generator more exposed to a
saturated critic's near-zero gradient. This matches the sweep's original selection-criteria
philosophy (`GENERATOR_GRADIENT_FIX.md` Stage 3): a candidate that merely *looks* fine on
final-epoch numbers isn't the same as one with healthy underlying training dynamics — and at
the longer 15-epoch horizon, `0.05`'s gradient health became the deciding factor Phase 1's
shorter horizon couldn't yet reveal.

### Conclusion (Phase 2 / overall)

**Final recommendation: `lambda_kl=0.05`** for the next full-scale training run — supersedes
Phase 1's provisional `0.005` pick. `lambda_kl=0` remains confirmed unsuitable (Phase 1).
`0.01` and `0.1` were not carried into Phase 2 and remain untested at the longer horizon; given
`0.05`'s clear margin and healthy gradient dynamics at 15 epochs, there's no signal motivating
a revisit of either.

**Before using this checkpoint anywhere:** Phase 2's `lambda_kl=0.05` run has no saved model
weights (see "Phase 2 Results" above — MN5 disk quota exceeded during the final checkpoint
write). Free space in `gpfs_projects` (deleted the stale pre-gradient-fix grid-search
checkpoints on 2026-07-26) has already been confirmed adequate; a fresh run with the same
config (`lambda_kl=0.05`, `n_critic=4`, `lr_gen=5e-6`, `lr_critic=5e-5`, `temperature=1.0`,
15+ epochs, full dataset) is needed to actually produce a usable checkpoint before generation
or further evaluation can proceed.
