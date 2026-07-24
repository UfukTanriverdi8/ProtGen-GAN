# Generator Gradient Flow Fix

> Companion doc to `CLAUDE.md`'s "ARCHITECTURAL ISSUE" section. This file holds the
> full research synthesis and staged implementation plan so `CLAUDE.md` can stay
> short. Read this before touching `models.py`, `loss.py`, or either training script
> for this fix.

---

## The problem

The generator's `.generate()` method returns discrete `torch.long` token IDs. The
generator-update step in both training scripts looks like this:

```python
fake_data = generate_fakes_for_batch(
    generator, tokenizer,
    input_ids_gen, attn_mask_gen,
    initial_masking_rate, iteration_fill_rate,
    min_temp, max_temp
)

gen_optimizer.zero_grad()
fake_scores = critic(fake_data, attention_mask=(fake_data != tokenizer.pad_token_id).long())
g_loss      = generator_loss(fake_scores)
g_loss.backward()
gen_optimizer.step()
```

`critic(fake_data, ...)` begins with an embedding lookup, `embedding_matrix[fake_data]`.
That lookup is differentiable with respect to `embedding_matrix` (gradient scatter-adds
into whichever rows were used, which is why the critic trains correctly) but **not**
differentiable with respect to the integer indices themselves — there is no derivative
of "which row did you pick." `g_loss.backward()` therefore reaches the critic's
parameters and stops dead at that lookup. `gen_optimizer.step()` has been applying an
effectively zero-valued update to the generator's ProtBERT weights across every
training run to date (~60–70 runs, ~8 months). This is true for both the old argmax
sampling and the current `torch.multinomial` sampling — both produce discrete
integers, neither is differentiable.

---

## Research summary

Full deep-research output is archived separately; this is the distilled version with
the parts that actually changed the plan.

**The field's three solution families**, and where the original three candidates
(soft embeddings, Gumbel-Softmax, REINFORCE) sit:

| Approach | Family | Verdict for this project |
|---|---|---|
| Soft embeddings: `softmax(logits/T) @ E` | Continuous relaxation | **Chosen.** Lowest implementation cost, lowest variance, second-order bias only under concentrated (peaked) logits — which is what a fine-tuned transformer produces. |
| Gumbel-Softmax: `gumbel_softmax(logits/T, tau) @ E` | Continuous relaxation | Workable fallback. Adds a temperature-annealing hyperparameter on top of WGAN-GP's own sensitivity; `hard=True` variant has first-order bias. No clear advantage over soft embeddings here. |
| REINFORCE / policy gradient | RL / policy gradient | **Rejected for the adversarial term.** WGAN-GP is already unstable; REINFORCE adds high-variance score-function gradients on top of it. Only justified if a *non-differentiable* reward (e.g. an external structure oracle) is added later — keep that as a separate, additional loss term, not a replacement for the differentiable critic term. |
| Straight-through estimator (hard forward / soft backward) | Continuous relaxation (4th family, added during research) | **Adopted for intermediate refinement steps.** Forward pass uses true one-hot (matches real data, keeps ProtBERT in-distribution); backward pass uses the soft surrogate gradient. First-order biased, but solves the input-type-mismatch problem for free. |

**Protein-specific precedent — DRAKES** (Wang et al., ICLR 2025): fine-tunes a masked
discrete-diffusion protein generator — architecturally close to this project's
iterative masked-refinement loop — by backpropagating a reward through the sampling
trajectory via Gumbel-Softmax, with **truncated backprop** through only the later
steps for tractability. Their **no-KL-anchor ablation collapsed**: percent of
sequences with positive predicted stability hit 100%, but median scRMSD (fold
fidelity) exploded from 0.918 to 7.307 — i.e. the generator learned to fool the reward
model while producing sequences that no longer fold into anything real. Direct,
quantitative warning for this project once the generator starts receiving real
gradient: an anchor to the pretrained model is not optional.

**Soft embeddings as the current best relaxation — Soft-Di[M]O** (Zhu, Wang,
Lathuilière, Kalogeiton; arXiv:2509.22925, Sept 2025): introduces the same soft
embedding mechanism as Option 1 for GAN-based refinement of a masked-diffusion
generator, reports smoother GAN training and lower-variance discriminator logits
versus Gumbel straight-through, and proves soft embeddings carry only second-order
bias under concentrated logits (vs. Gumbel-Softmax ST's first-order bias). Also
reports **mode collapse at high initial mask ratio (0.95) under GAN-only training** —
directly relevant to this project's blind mode (100%-masked start).

**ProteinGAN** (Repecka et al., *Nature Machine Intelligence* 2021): WGAN-style
protein-sequence generator, produced enzymes where 24% of tested sequences were
soluble and catalytically active — establishes that a GAN *can* generate functional
enzymes, supporting the overall approach rather than any specific gradient-flow fix.

**The detail missing from the original candidate list — WGAN-GP input-type
mismatch.** Multiple text-GAN papers (TextKD-GAN, arXiv:1905.01976; Soft-GAN/LATEXT-GAN,
Haidar et al., NAACL-HLT 2019) document that if the critic sees one-hot real sequences
alongside softmax (dense) fake sequences, it learns the trivial "is this sparse?"
feature and the Wasserstein gradient vanishes. **This means the fix is not just "make
fake_data continuous" — real sequences must also be routed through the embedding
matrix, and the WGAN-GP gradient penalty must interpolate in embedding space, not on
raw integer IDs.** Skipping this turns a zero-gradient bug into a near-zero-gradient
bug that would look like progress in the loss curves without actually training
anything useful.

**What the iterative refinement loop makes uniquely hard** (this project is *not*
autoregressive, which is mostly an advantage, but the multi-step mask-and-refill
structure has its own wrinkle): discreteness shows up at two points — the
intermediate refinement steps, where committed tokens get re-fed as input to the next
step, and the final critic-facing output. Gradient only reaches early-step logits if
every intermediate sampling step on the differentiable path is relaxed too. Relaxing
only the final output leaves gradient unable to flow past the last step into the
weights that produced earlier fills. The accepted compromise (per DRAKES and DRaFT,
Clark et al. ICLR 2024) is **truncated backprop through only the final K steps**,
starting at K=1, with straight-through (hard forward) at the intermediate commits so
ProtBERT keeps seeing genuinely discrete input at every step except the one that
actually gets gradient — also keeping training closer to how it was originally
fine-tuned.

---

## Decision

Use **soft embeddings** (not Gumbel-Softmax, not REINFORCE) for the critic-facing
output, **straight-through** at intermediate refinement-step commits, **truncate
backprop to K=1** to start, **embed real sequences through the same matrix** and
**interpolate the WGAN-GP gradient penalty in embedding space**, and **add a KL or
MLM anchor** to the frozen pretrained ProtBERT to prevent the DRAKES-style collapse.
Validate in **seeded mode first** — blind mode's full masking plus adversarial loss
is a documented mode-collapse risk (Soft-Di[M]O, high-mask-ratio finding) and should
not be the first test of this fix.

This is not "it depends" at the family level — a differentiable relaxation is the
right choice because the critic is differentiable and REINFORCE would discard usable
gradient. It *is* "try X first, fall back to Y" within that family: soft embeddings
first; switch to straight-through at the critic-facing step too if the critic
saturates on the real/fake input-type mismatch even after embedding the real
sequences correctly.

---

## Staged implementation plan

**Stage 0 ✅ — Instrument before touching anything.**
Log the generator's ProtBERT gradient norm immediately after `g_loss.backward()` in
both training scripts. Confirmed ≈0 before fix across all prior runs. This is the
before/after sanity check.

**Stage 1 ✅ — Soft embeddings + embed-the-real + K=1 truncated backprop.**
(Note: the original plan split this into Stage 1 and Stage 2. They were implemented
together — K=1 is implicit in `compute_soft_embeds`, which does one forward pass on a
re-masked copy of the completed sequence and does not backprop through the iterative
fill loop.)
- `compute_soft_embeds()` in `models.py`: re-masks 50% of the completed sequence
  (special tokens excluded), runs one generator forward pass on that masked input →
  `softmax(logits/T) @ critic_word_weight`, then blends soft embeds at the remasked
  positions with hard original-token embeds elsewhere → full embeds via `inputs_embeds`.
  Gradient reaches the generator only through the remasked slots. K=1 by design — the
  fill loop that produced `input_ids` is not in this graph.
- Both real and fake embedded via `critic.protbert.bert.embeddings()` before critic
  update so the critic cannot distinguish real/fake by embedding sparsity.
- `compute_gradient_penalty()` now accepts pre-computed `[B,L,H]` embedding tensors
  and interpolates in embedding space.

*Observed (2026-06-26, run koqatr9h):* `gen_grad_norm` confirmed > 0 in epoch 2+
for the first time across all training history. Generator is receiving adversarial
signal. `unique_ratio` held at 1.0. Training stable.

**Stage 2 ✅ — KL anchor against frozen reference ProtBERT.**
`compute_kl_anchor()` in `models.py`: `KL(generator || frozen_ref)` at same temperature
as `compute_soft_embeds`, scored only at the remasked positions where the generator made
a genuine prediction. `ref_protbert` loaded fp16, frozen, eval mode — never updated, and
fed the *same* masked input the generator saw. `g_loss = -critic(soft_embeds).mean() +
lambda_kl * KL(gen || ref)` (default `lambda_kl=0.01`). `kl_loss` logged to wandb.

*Observed (2026-06-27, run ouqv9wox, 5 epochs):* `gen_grad_norm` > 0 confirmed again.
`kl_loss` oscillated wildly 89–1236 — root cause was the out-of-distribution input, now
fixed (see below). Re-validate on the next seeded run.

**✅ FIXED — KL anchor now computed on a re-masked input (2026-06-27, commit `c18c89a`):**
The old `compute_soft_embeds` received `fake_data` after the iterative fill loop — fully
revealed, zero [MASK] tokens. ProtBERT is MLM-trained and had never seen fully-revealed
input, so its logits were uncalibrated; even small adversarial weight updates caused large
logit swings on that out-of-distribution input → KL exploded. The clamp
(`gen_probs.clamp(min=1e-8)`) and `clip_grad_norm_(max_norm=1.0)` prevented NaN crashes and
weight blow-up but could not stabilise the loss itself.

**The fix:** `compute_soft_embeds` now re-masks 50% of the completed sequence (special
tokens excluded) and runs the generator on that masked input, so ProtBERT stays
in-distribution. The soft path feeds the critic only at the remasked positions (hard
original-token embeds elsewhere); `compute_kl_anchor` evaluates the frozen reference on the
identical masked input and scores KL only over those remasked positions. Both training
scripts updated. Expectation: `kl_loss` changes smoothly under weight updates — confirm on
the next seeded run.

**Stage 3 ✅ — Validate in seeded mode before blind mode.**
(Originally Stage 4.) Seeded mode partially validated (ouqv9wox), but under the OOD KL bug.
BUG 4 (attention mask excluding [MASK] tokens in blind mode) was fixed separately — no
longer a blocker. With the KL fix in place, re-run seeded mode and confirm stable
`kl_loss` before promoting to blind mode.

*Observed (2026-06-27, post-`c18c89a`, both 3 epochs, seeded mode, n_critic=4):*

- **`xwres3cz` (`test-10p-soft-nc4-kl0`, lambda_kl=0):** `gen_grad_norm` > 0 from epoch 2
  on, bounded ~0.1–30 (vs. 100k+ pre-fix) — no `kl_loss` logged since `ref_protbert` is
  skipped entirely at `lambda_kl=0`. `unique_ratio` held at 1.0 throughout, but sequence
  *quality* collapsed hard without the anchor: `plddt` 0.74→0.50→0.36, `scAccuracy`
  0.38→0.22→0.03, `progres` 0.91→0.59→0.50, `pairwise_tm` 0.76→0.28→0.20 from baseline to
  end of epoch 3. Diverse but increasingly non-protein-like — confirms the KL anchor is
  necessary, not just noise.
- **`ydsjq9f4` (`test-10p-soft-nc4-kl1e2`, lambda_kl=0.01):** `gen_grad_norm` > 0 from
  epoch 2 on, bounded ~0.3–2.4. `kl_loss` ≈0 in epoch 1 (frozen, generator = reference),
  then **oscillates in a bounded 0–4 range** through epochs 2–3 — three orders of
  magnitude smaller than the pre-fix 89–1236 range, and never explodes. `unique_ratio`
  held at 1.0, and quality metrics stayed healthy: `plddt` 0.75→0.69→0.70, `scAccuracy`
  improved 0.38→0.40→0.42, `progres` 0.92→0.87, `pairwise_tm` 0.75→0.64.

**Conclusion: the fix works.** `kl_loss` no longer oscillates wildly, `gen_grad_norm > 0`
in both configurations, and `unique_ratio` never collapses. The lambda_kl=0 run additionally
shows the KL anchor is load-bearing for sequence quality (not diversity) — without it the
critic score keeps improving while structural plausibility craters. Neither run was ever
analyzed in a Claude Code session at the time; recovered from wandb run history on
2026-07-22.

**✅ Independent correctness check (2026-07-22, `tests/check_kl_identity.py`):** the
observations above only show `kl_loss` *behaving* plausibly during real training — they
don't prove `compute_kl_anchor` itself is bug-free (wrong tensor passed to the wrong
argument, misaligned `remask_positions`, a skipped softmax could all still produce a
smooth-looking curve). Added a standalone script that loads the checkpoint twice — once
as `generator`, once as `ref_protbert` — and confirms the mathematical identity
`KL(P‖P) = 0`: with identical weights on the same masked input, `kl_loss = 5.1e-7`
(~0, as required). A negative control with the reference weights perturbed gives
`kl_loss = 12.9` (clearly nonzero), ruling out a test that trivially "passes" because
`ref_protbert`'s output isn't actually being used. Also confirmed
`tokenizer.mask_token_id` is consistent everywhere it's used (`= 4` for this checkpoint's
vocab) — `generate.py` previously relied on a hardcoded default matching this by
coincidence rather than reading it from the tokenizer; fixed in commit `718e446`.

**Stage 4 (only if needed) — Non-differentiable reward via PPO/REINFORCE.**
If a non-differentiable signal is added later (e.g. an external structure or function
oracle), add it as a *separate* PPO-style term with its own baseline and KL constraint,
on top of the differentiable critic term — don't replace the critic term with it.

---

## Validation criteria / failure tripwires

- **Generator gradient norm stays ≈0 after Stage 1** → the soft-embedding path isn't
  wired into the differentiable graph; check that the critic call in the generator
  update step uses `soft_embeds`, not hard-sampled `fake_data`.
  *Observed: gen_grad_norm confirmed > 0 (epoch 2+) in both koqatr9h and ouqv9wox.
  Early zeros at epoch 1 are expected — ProtBERT is frozen then.*
- **Critic real/fake logit gap stays near-saturated** → real and fake still look
  different in kind to the critic; check that real sequences are being embedded
  through the same matrix, not passed as raw IDs.
- **kl_loss oscillates wildly (89–1236 range observed in ouqv9wox)** → this was the
  out-of-distribution input issue, fixed in `c18c89a` (`compute_soft_embeds` now operates
  on a re-masked input). **Confirmed resolved**: post-fix run `ydsjq9f4` (lambda_kl=0.01,
  3 epochs) shows `kl_loss` bounded to a 0–4 range throughout. If oscillation reappears,
  fall back to `--lambda_kl 0.0` to isolate clean adversarial dynamics and investigate
  further.
- **Oscillating critic loss / generator collapsing (unique_ratio < 1.0)** → confirm
  KL anchor is active (or increase lambda_kl), reduce n_critic, or fall back to seeded
  mode if testing blind mode.
- **Memory blows up** → stay at K=1; add gradient checkpointing only if K is increased.
- **Sequences score well against the critic but fail structural/biophysical checks
  downstream** (the protein analogue of the DRAKES scRMSD blow-up) → reward hacking,
  not a gradient-flow problem. Strengthen the KL anchor (increase lambda_kl), don't
  touch the gradient path.

---

## Open risks / caveats

- No published paper matches this project's exact configuration (ProtBERT generator
  + ProtBERT critic + WGAN-GP + iterative masked refinement). The relaxation
  mechanism is well-supported by adjacent literature (text-GAN, protein diffusion,
  image-domain GAN distillation) but expect real tuning, not a drop-in recipe.
- DRAKES and Soft-Di[M]O are not WGAN-GP studies (DRAKES is reward fine-tuning of
  diffusion; Soft-Di[M]O is GAN-based distillation of a one-step generator). The
  transfer of their mechanisms to this WGAN-GP loop is sound but unverified in
  exactly this combination.
- MDNS (2025) reports DRAKES can mislearn a sampler when backpropagating through many
  relaxed trajectory steps without truncation — reinforces starting at K=1 rather
  than backpropagating through the full refinement trajectory.
- The amino-acid vocabulary is small (~25–30 tokens vs. tens of thousands of word
  tokens in NLP literature), which should make relaxation easier here, not harder,
  but this is an inference from vocabulary size rather than a directly verified
  protein-specific result.
- **Confidence-inflation shortcut, not fully closed by the KL anchor (2026-07-03).**
  The critic only ever trains on hard (fully discrete) embeddings — both real and fake
  — via the embed-the-real fix. At generator-update time it's asked to score a blend
  that's 50% hard / 50% soft (`compute_soft_embeds`, `models.py:159`), where the soft
  half is a probability-weighted average over the embedding matrix. The cheapest way
  for the generator to raise its critic score isn't necessarily "produce a more
  DNMT-like sequence" — it's to make its own output logits more peaked/confident at
  the remasked positions, which mechanically pulls `soft_sequence` closer to a hard
  one-hot embedding regardless of *which* token it's confident about. This is a
  distribution-shift/reward-hacking risk distinct from the DRAKES-style collapse the
  KL anchor already targets.
  - The KL anchor (`compute_kl_anchor`) partially guards against this: sharpening
    *away* from the frozen reference's distribution at a remasked position raises KL
    and is penalized. But where the frozen reference is already confident at a given
    position (common for MLM models on "easy" positions), sharpening toward that same
    peak costs ~0 KL — so the exploit isn't blocked exactly where it would be
    cheapest to pull off.
  - **This failure mode is quiet.** Unlike gradient instability (NaNs, exploding
    losses) or KL oscillation, confidence inflation wouldn't spike any metric
    currently logged (`kl_loss`, `gen_grad_norm`, `unique_ratio`). It would just
    slowly waste generator capacity on looking-more-certain rather than being-more-real.
  - **Proposed diagnostic (not yet implemented):** log the average max-probability
    (or entropy) of `gen_probs` at the remasked positions returned by
    `compute_soft_embeds`, per generator-update step, in both training scripts. A
    climbing average max-probability / falling entropy over training **without** a
    corresponding improvement in downstream quality metrics (pLDDT, scAccuracy,
    unique_ratio) would be the signature of this exploit. If observed, the fallback is
    the same one already named for the input-type mismatch: switch to a
    straight-through estimator at the critic-facing step (hard forward pass, soft
    gradient backward) so the critic never scores a blurred input to begin with.

---

## References

- Wang et al., "Fine-Tuning Discrete Diffusion Models via Reward Optimization with
  Applications to DNA and Protein Design" (DRAKES), arXiv:2410.13643, ICLR 2025
- Zhu, Wang, Lathuilière, Kalogeiton, Soft-Di[M]O, arXiv:2509.22925, Sept 2025
- Repecka et al., "Expanding functional protein sequence space using generative
  adversarial networks" (ProteinGAN), *Nature Machine Intelligence* 3, 324–333, 2021
- Haidar, Rezagholizadeh, Do-Omri, Rashid, "Latent Code and Text-based Generative
  Adversarial Networks for Soft-text Generation" (LATEXT-GAN), NAACL-HLT 2019,
  arXiv:1904.07293
- Haidar & Rezagholizadeh, TextKD-GAN, arXiv:1905.01976
- Yu et al., SeqGAN, AAAI 2017
- Nie et al., RelGAN, ICLR 2019
- de Masson d'Autume et al., ScratchGAN, NeurIPS 2019
- Clark et al., DRaFT, ICLR 2024
- MDNS, arXiv:2508.10684, 2025
