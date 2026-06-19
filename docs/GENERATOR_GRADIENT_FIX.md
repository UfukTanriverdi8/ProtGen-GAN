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

**Stage 0 — Instrument before touching anything.**
Log the generator's ProtBERT gradient norm immediately after `g_loss.backward()` in
both training scripts. Should currently print ≈0. This is the before/after sanity
check — confirms the bug, and later confirms the fix.

**Stage 1 — Soft embeddings + embed-the-real.**
- `models.py` / `generate_fakes_for_batch`: add a critic-facing path that returns
  `soft_embeds = F.softmax(logits / T, dim=-1) @ critic.embedding_matrix` instead of
  hard IDs. Keep hard sampling for the actual output sequences used everywhere else
  (logging, FASTA export, validity checks, evaluation).
- Critic: route **real sequences** through the same embedding matrix instead of raw
  IDs, using the existing 3D-embedding branch already present in `models.py`.
- `loss.py` `compute_gradient_penalty`: interpolate in embedding space —
  `x_hat = alpha * real_embeds + (1 - alpha) * soft_embeds`, penalize
  `(||grad_{x_hat} critic(x_hat)||_2 - 1) ** 2`.
- Generator loss: `g_loss = -critic(soft_embeds).mean()`.
- Start temperature near T=1; watch the real/fake critic-logit gap. If the critic
  still separates trivially after this stage, the input-mismatch fix is incomplete.

**Stage 2 — Truncate backprop to K=1.**
Only the final refinement step gets gradient initially. Intermediate commits stay
straight-through (hard forward / soft backward) so ProtBERT stays in-distribution.
Increase K only if the adversarial signal proves too weak, adding gradient
checkpointing if memory becomes a problem.

**Stage 3 — Add the anchor.**
A KL term against the frozen pretrained ProtBERT, or a simpler MLM cross-entropy
penalty, weighted into the generator loss. This is the direct mitigation for the
DRAKES no-KL collapse (scRMSD 0.918 → 7.307).

**Stage 4 — Validate in seeded mode before blind mode.**
Confirm the fix produces sane training dynamics in seeded mode first. Blind mode also
still has BUG 4 (attention mask excludes `[MASK]` tokens) unfixed as of this writing —
fix that before testing blind mode with the new gradient path, or the two issues will
be tangled together in any debugging.

**Stage 5 (only if needed) — Non-differentiable reward via PPO/REINFORCE.**
If a non-differentiable signal is added later (e.g. an external structure or function
oracle that can't be backpropagated through), add it as a *separate* PPO-style term
with its own baseline and KL constraint, on top of the differentiable critic term —
don't replace the critic term with it.

---

## Validation criteria / failure tripwires

- **Generator gradient norm stays ≈0 after Stage 1** → the soft-embedding path isn't
  actually wired into the differentiable graph; check that `fake_data` used for the
  critic call is the soft-embeds path, not the hard-sampled path.
- **Critic real/fake logit gap stays near-saturated** → real and fake still look
  different in kind to the critic; check that real sequences are being embedded
  through the same matrix, not passed as raw IDs.
- **Oscillating critic loss / generator collapsing to a handful of sequences** → slow
  the temperature annealing, confirm the Stage 3 anchor is active, consider reducing
  the critic update ratio, or fall back to seeded mode if testing blind mode.
- **Memory blows up when K is increased** → add gradient checkpointing; if still
  infeasible, stay at K=1.
- **Sequences score well against the critic but fail structural/biophysical checks
  downstream** (the protein analogue of the DRAKES scRMSD blow-up) → this is reward
  hacking, not a gradient-flow problem. Strengthen the anchor, don't touch the
  gradient path.

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
