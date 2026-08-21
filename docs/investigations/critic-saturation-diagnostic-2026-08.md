# Critic-Saturation Diagnostic

**Started:** 2026-08-20
**Question:** Does `critic_loss` saturating at exactly `5.0000` (CLAUDE.md item 20) reflect
genuine critic convergence, or degenerate collapse (the critic becoming a constant function
of its input) — and is whichever answer holds universal/architectural, or dependent on
`lambda_kl`?
**Status:** Core question answered. One mechanism (the epoch-12 instability in the
`lambda_kl=0.005` arm) remains only partially understood — flagged as a follow-up, not blocking.

## Design

The lambda_kl sweep (`docs/sweeps/lambda-kl-sweep-2026-07.md`) picked `lambda_kl=0.05` as the
winner using `gen_grad_norm`/quality metrics measured while `critic_loss` was already pinned
at `5.0000` in most arms — at the time, nobody could tell whether that pinning meant the
critic had genuinely converged (still discriminative, scores just stopped moving in aggregate)
or had degenerately collapsed (stopped being a function of its input at all, real and fake
alike mapping to the same scalar). If it's the latter, any part of the original winner
selection that leaned on critic-derived signal during that window is standing on an unreliable
foundation.

This became answerable once held-out critic validation and finer-grained loss/score logging
landed in `10p_train.py` (`--holdout_size`, `--loss_log_every`, item 22) — per-example score
variance (`*_score_std`) is what actually distinguishes the two explanations, since both
produce the same `critic_loss` aggregate.

Testing only `lambda_kl=0.05` would not have been enough: it couldn't distinguish "collapse is
universal/architectural, unrelated to `lambda_kl`" from "collapse is `lambda_kl`-dependent, and
the original sweep conclusion holds on firmer ground." Both of the sweep's Phase 2 finalists
(`lambda_kl=0.005` and `lambda_kl=0.05`) were re-run side by side, using the same fixed
hyperparameters as the original sweep (`n_critic=4`, `lambda_gp=5`, `lr_gen=5e-6`,
`lr_critic=5e-5`, `temperature=1.0`, `seed=89`, 15 epochs, `batch_size=8`), with
`--holdout_size 200` and `--loss_log_every 25` newly enabled.

---

## Runs involved

| Run | Purpose in this investigation | File |
|---|---|---|
| critic-saturation-diagnostic-kl0.005 (6v24i366) | lambda_kl=0.005 arm | docs/runs/critic-saturation-diagnostic-kl0.005_6v24i366.md |
| critic-saturation-diagnostic-kl0.05 (19y835y0) | lambda_kl=0.05 arm | docs/runs/critic-saturation-diagnostic-kl0.05_19y835y0.md |

## Findings

**Why `critic_loss` pins at exactly `5.0000`, mechanistically.** `loss.py`'s critic loss is
`-mean(real_scores) + mean(fake_scores) + lambda_gp * gradient_penalty`. If the critic
collapses to a constant function of its input, `mean(real_scores) ≈ mean(fake_scores)`, so the
first two terms cancel to ≈0. A constant function also has zero gradient with respect to its
input everywhere, so `gradient_penalty = mean((‖∇‖ - 1)²)` evaluates to `(0 - 1)² = 1`. With
`lambda_gp=5` in this diagnostic, that leaves `critic_loss ≈ 0 + 5·1 = 5.0000` — not an
emergent equilibrium, but the loss formula's arithmetic signature of a critic that has stopped
reacting to its input at all.

**Both arms show genuine degenerate collapse, not benign convergence.** In both runs,
`train_real_score_std` and `holdout_real_score_std` collapse to the ~1e-5–1e-6 order of
magnitude in lockstep with `critic_loss` pinning near 5.0000 (kl=0.05: from epoch 4–5 onward,
persisting to the end of the run; kl=0.005: epoch 2 through epoch 11). This is the per-example
variance signature that distinguishes "the critic ignores its input" from "the critic still
discriminates, its aggregate just stopped moving" — and it shows up on `holdout_real_score_std`
too, computed on sequences the critic never trained on, which rules out training-set
memorization as the explanation.

**The collapse itself is universal, not `lambda_kl`-dependent.** `lambda_kl` only appears in
the generator's loss (the KL anchor term) — it never appears anywhere in `critic_loss`. Both
arms collapse via the identical mechanism regardless of `lambda_kl`'s value, consistent with
this: whether collapse happens looks architectural (most plausibly tied to `lambda_gp`'s
contribution being cheap to satisfy via a constant output), not a consequence of the swept
parameter.

**What `lambda_kl` *does* change is what happens around the collapse.** The two arms diverge
sharply here:

- **`lambda_kl=0.05`**: stays in a flat, silent, persistently-collapsed state from epoch 5 all
  the way to epoch 15 (final `critic_loss` 4.99932, barely drifted). `gen_grad_norm`/`kl_loss`
  stay in their normal bounded bands the entire run, and `generator_loss` stays flat in a
  narrow -0.06 to -0.08 band throughout. The only anomaly in this arm at all: a single
  `critic_loss` spike to 165.7 at the epoch-2 checkpoint, resolved by epoch 3, unrelated to
  anything below.
- **`lambda_kl=0.005`**: after the same initial collapse (epoch 2–11), a real instability
  episode starts around epoch 12 (step ~6230) and is *not* present anywhere in the 0.05 arm.
  Onset-to-resolution timeline (raw full-resolution history, not epoch-boundary snapshots —
  the whole episode starts and mostly resolves inside two epochs):
  - `gen_grad_norm`: onset ~step 6233 (baseline ~0.1–0.5), climbs sharply, peaks at **116** at
    step 6647, resolved back to baseline (~0.1–0.4) by ~step 6900.
  - `kl_loss`: rises from step ~6240, peaks at **3.77** (step 6510, vs. ~0.1–0.3 baseline),
    settles by end of run to a new, higher-than-before baseline (~0.7–1.5) rather than
    returning fully to its pre-event level.
  - `critic_loss_step`: dips to a minimum of **4.44** at step 6458, recovers back to ~5.0 by
    step ~6918.
  - `train_real_score_std`: spikes to **0.0486** at step 6431, decays back to baseline
    (~0.002) by step ~6996.
  - `train_fake_score_std`: spikes to **0.0325**, but notably ~330 steps *later* than the real
    score std (peak at step 6816 vs. 6431) — decays back to baseline by step ~7149.
  - `holdout_real_score_std`: spikes to **0.0638** at step 6457, essentially simultaneous with
    the training-pool real-score spike — meaning the critic's own weights are moving during
    this window, not merely reacting to a shifted distribution of generated fakes.
  - A second, smaller echo of the same pattern recurs around step 7250–7300 before everything
    settles for the rest of the run.
  - **`generator_loss` does not recover.** Unlike every signal above, it climbs across four
    consecutive epoch-end checkpoints and never comes back down: epoch 11 (-0.073, healthy) →
    epoch 12 (-0.770) → epoch 13 (-1.644) → epoch 14 (+3.862) → epoch 15/final (+5.937), a sign
    flip that persists to the very end of training even while the training-loop signals above
    have gone quiet again.

**Ordering hints at a plausible (unconfirmed) mechanism.** The generator-side signals
(`gen_grad_norm`, `kl_loss`) start moving first (~step 6230–6250); the critic-side signals
(`critic_loss`, score-stds) follow ~100–200 steps later (~step 6330–6430). One hypothesis: a
collapsed critic gives the generator essentially zero adversarial gradient too (since
`generator_loss = -mean(fake_scores)` is uninformative if `fake_scores` are constant), so both
sides have been sitting in a near-zero-gradient regime for ~10 epochs. AdamW's second-moment
estimate would decay during that stretch, so when any real gradient signal reappears, the
normalized step gets amplified into an outsized update on the generator's weights — which then
produces genuinely different-looking fake sequences, "waking" the critic out of collapse by
confronting it with novel input for the first time in epochs. This is plausible and consistent
with the ordering, but **not independently confirmed** (e.g. by inspecting AdamW's `v` buffer
directly) — flagged as an open question, not a settled finding.

**Structural quality metrics show no relationship to any of this.** `plddt`, `scAccuracy`,
`progres`, `pairwise_tm`, and `unique_ratio` stay flat/noisy in both arms across the whole run,
uncorrelated with the saturation state, the epoch-12 event, or the `generator_loss` trend —
consistent with CLAUDE.md's existing finding that these metrics plateau early regardless of
what the adversarial dynamics are doing.

**No held-out fake-score data for either arm.** `--holdout_eval_fakes` was not set on either
run, so real/fake symmetry on the held-out set could not be checked here.

## Answer

**Genuine degenerate collapse, confirmed in both arms.** The std-collapse signature, present on
both training-pool and held-out real scores, rules out benign convergence. The collapse
mechanism (tied to `lambda_gp`'s contribution to `critic_loss`, never `lambda_kl`) looks
architectural rather than `lambda_kl`-dependent — both arms collapse via the identical pathway.

**What *is* `lambda_kl`-dependent is downstream stability, not whether collapse occurs.**
`lambda_kl=0.05` produces a boring, stable, permanently-collapsed critic for the entire run.
`lambda_kl=0.005` produces the same initial collapse, then a genuine (partially unexplained)
instability episode around epoch 12 whose effects — specifically on `generator_loss` — never
resolve by the end of training.

**Implication for the original "0.05 wins" sweep conclusion**
(`docs/sweeps/lambda-kl-sweep-2026-07.md`): that conclusion was built partly on
`gen_grad_norm`/critic-derived signal measured while the critic was already collapsed in most
arms — this diagnostic confirms that signal really was degenerate during the measurement
window, in both re-tested arms. So the part of the original comparison that leaned on
critic-loss/`gen_grad_norm` dynamics is on weaker ground than assumed. The comparison's more
defensible legs are the structural quality metrics (which stayed flat and broadly comparable
between arms here too) and — newly, from this diagnostic — downstream stability: `0.05`'s flat,
uneventful trajectory versus `0.005`'s disruptive, still-partially-unresolved episode. `0.05`
isn't disproven as the better choice, but the evidentiary basis for preferring it should now
rest on those two legs, not on the original critic-derived dynamics.

## Next steps

- Test whether `lambda_gp=5` is too strong relative to the Wasserstein term's natural scale,
  making "the critic ignores everything" an artificially cheap optimum — try a smaller
  `lambda_gp` and check whether collapse still occurs.
- Directly test the AdamW post-flat-region-overshoot hypothesis for the epoch-12 event (e.g.
  log/inspect optimizer second-moment statistics, or compare against a different optimizer/eps
  setting) rather than leaving it as an inference from ordering alone.
- Re-run with `--holdout_eval_fakes` set to get real/fake symmetry on the held-out set.
- Check whether other `lambda_kl` values reproduce the same pattern (universal collapse,
  `lambda_kl`-dependent post-collapse stability) to see how far it generalizes beyond these two
  arms.
- Update CLAUDE.md item 20 with this investigation's answer (separate step).
