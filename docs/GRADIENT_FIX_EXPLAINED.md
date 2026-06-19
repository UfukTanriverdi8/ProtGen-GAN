# Understanding the Generator Gradient Fix

> Conceptual companion to `GENERATOR_GRADIENT_FIX.md` (which holds the staged
> implementation plan and research citations). This document explains **why** the fix
> works, from first principles, without assuming prior knowledge of differentiable
> relaxation techniques.

---

## Table of Contents

1. [How PyTorch Autograd Works (the 2-minute version)](#1-how-pytorch-autograd-works)
2. [Where the Gradient Dies in ProtGen](#2-where-the-gradient-dies-in-protgen)
3. [What "Soft Embeddings" Actually Means](#3-what-soft-embeddings-actually-means)
4. [Why Real Sequences Must Be Embedded Too](#4-why-real-sequences-must-be-embedded-too)
5. [The Multi-Step Problem and Straight-Through](#5-the-multi-step-problem-and-straight-through)
6. [Why We Truncate to K=1](#6-why-we-truncate-to-k1)
7. [The KL Anchor — Preventing Reward Hacking](#7-the-kl-anchor--preventing-reward-hacking)
8. [The Gradient Penalty in Embedding Space](#8-the-gradient-penalty-in-embedding-space)
9. [Putting It All Together](#9-putting-it-all-together)

---

## 1. How PyTorch Autograd Works

PyTorch builds a computation graph as you do math on tensors. Every operation (add,
multiply, matmul, softmax, ...) records what inputs it used and what output it produced.
When you call `loss.backward()`, PyTorch walks this graph in reverse, computing the
derivative of the loss with respect to every tensor that contributed to it. These
derivatives are called **gradients**.

Gradients answer one question: "If I nudge this value slightly, how does the loss change?"

The optimizer (`AdamW`, `SGD`, etc.) then uses these gradients to update the model's
weights — pushing them in the direction that reduces the loss.

**Key rule: autograd only works on floating-point tensors, and only through differentiable
operations.** If any link in the chain is non-differentiable, the gradient cannot pass
through it, and everything upstream of that break receives zero gradient.

---

## 2. Where the Gradient Dies in ProtGen

### The intended flow (how a GAN is supposed to work)

```
Generator produces fake data
        ↓
Critic scores fake data
        ↓
Compute generator loss from the critic's score
        ↓
loss.backward() sends gradients back through:
    Critic layers  ←  gradients flow fine
        ↓
    Generator layers  ←  generator learns to fool the critic
```

### What actually happens in ProtGen

```
Generator (ProtBERT) produces logits  →  [batch, seq_len, vocab_size] float tensor
        ↓
torch.multinomial (or argmax) samples token IDs  →  [batch, seq_len] integer tensor (torch.long)
        ↓  ← ══════ GRADIENT WALL ══════
Critic receives integer token IDs
        ↓
nn.Embedding lookup: embedding_matrix[token_id]  →  float tensor again
        ↓
Critic transformer layers + classification head → score
        ↓
g_loss = -score.mean()
        ↓
g_loss.backward() flows gradients through:
    Classification head  ←  ✓ gets gradient
    Critic transformer   ←  ✓ gets gradient
    nn.Embedding matrix  ←  ✓ gets gradient (scatter-add into used rows)
    The integer indices   ←  ✗ DEAD — integers have no gradient
    Generator (ProtBERT) ←  ✗ DEAD — everything upstream of the wall is unreachable
```

### Why exactly can't integers have gradients?

A gradient is a derivative: "how does the output change when this input changes by an
infinitesimal amount?" Integers can't change by infinitesimal amounts — you can go from
5 to 6, but there's no 5.0001. The concept of a derivative doesn't apply.

More concretely, `nn.Embedding` is a lookup table. `embedding[5]` returns row 5.
PyTorch can compute how the loss changes if **the values in row 5** change (that's the
embedding matrix gradient). But it cannot compute how the loss changes if **the index
changes from 5 to 6** — that's a discrete jump to a completely different row, not a
smooth change.

### The consequence

The generator's ProtBERT weights have **never been updated by the adversarial signal**
in ~60–70 training runs over ~8 months. The only effect of `gen_optimizer.step()` has
been AdamW's weight decay, which slowly erodes the fine-tuned weights toward zero.

The "GAN" has been: a frozen fine-tuned ProtBERT generating sequences the same way it
always would, while the critic learns to classify against this static generator. The
adversarial game — where the generator improves based on critic feedback — has never
occurred.

---

## 3. What "Soft Embeddings" Actually Means

### The core idea

Instead of converting logits → discrete token → embedding lookup, we bypass the
discrete step entirely:

```
logits → softmax → probability distribution → weighted average of ALL embeddings
```

### Step by step

**Step 1: Generator produces logits** (this part doesn't change)

```python
logits = generator.forward(input_ids, attention_mask)
# logits shape: [batch, seq_len, vocab_size]
# e.g., [8, 350, 30]  (8 sequences, 350 positions, 30 amino acids)
```

Each value in `logits` represents how much the generator "wants" to place each amino
acid at each position. Higher logit = more preferred.

**Step 2 (OLD): Sample a discrete token**

```python
probs = F.softmax(logits / temperature, dim=-1)  # normalize to probabilities
token_id = torch.multinomial(probs, 1)            # sample one token per position
# token_id shape: [batch, seq_len] — integers like [4, 7, 12, 4, 19, ...]
```

This is where the gradient dies. `torch.multinomial` produces integers.

**Step 2 (NEW): Compute soft embeddings**

```python
probs = F.softmax(logits / temperature, dim=-1)
# probs shape: [batch, seq_len, vocab_size]
# e.g., one position might be: [0.70, 0.20, 0.05, 0.03, 0.01, 0.01, ...]
#                                Ala   Gly   Val   Leu   ...

soft_embeds = probs @ embedding_matrix
# probs:            [batch, seq_len, vocab_size]
# embedding_matrix: [vocab_size, hidden_dim]
# soft_embeds:      [batch, seq_len, hidden_dim]
```

### What does `probs @ embedding_matrix` actually compute?

For each position in the sequence, it computes a **weighted average of all amino acid
embeddings**, where the weights are the probabilities.

Example for one position where the generator is 70% confident it's Alanine:

```
soft_embed = 0.70 * embed(Ala) + 0.20 * embed(Gly) + 0.05 * embed(Val) + 0.03 * embed(Leu) + ...
```

Compare to the hard-token path:

```
hard_embed = embed(Ala)    # 100% Alanine, nothing else
```

The soft embedding is a "blurry" version — it represents the generator's uncertainty.
As the generator gets more confident, the probabilities sharpen toward one-hot and the
soft embedding converges toward the hard embedding.

### Why this fixes the gradient problem

Every operation in the new path is differentiable:

```
logits  →  softmax  →  matmul  →  critic  →  loss
  ✓          ✓          ✓         ✓          ✓
```

- `softmax` is differentiable (smooth function of logits)
- Matrix multiplication (`@`) is differentiable
- Everything in the critic is differentiable once it receives float input

So `loss.backward()` can now propagate gradients all the way from the critic's score,
through the matmul, through the softmax, back into the generator's logits, and from
there into ProtBERT's weights. The adversarial game can finally begin.

### Important: soft embeddings are only for the critic-facing path

The generator still produces hard tokens (via `torch.multinomial`) for everything else:
logging sequences, writing FASTA files, computing evaluation metrics, saving outputs.
You can't write a "70% Alanine" to a FASTA file — real sequences need discrete amino
acids.

The soft embedding path exists **only** during the generator's training step, so the
critic can provide a gradient signal to the generator.

---

## 4. Why Real Sequences Must Be Embedded Too

This is the subtle part that's easy to miss. If you only fix the fake data path:

```
Real sequences → nn.Embedding lookup → crisp single-vector per position
Fake sequences → softmax @ embedding  → blurry mixture of vectors per position
```

The critic can trivially distinguish them by **format**, not content. A simple
statistical check — "are these embeddings exactly one row from the table, or a weighted
mix?" — gives perfect classification without learning anything about protein quality.

It's like a forgery detector that just checks if the ink is still wet. It's trivially
correct but completely useless as a training signal. The gradients flowing back to the
generator would just say "make your outputs less blurry" rather than "make better
proteins."

### The fix

Pass real sequences through the same operation:

```python
# Real: convert to one-hot, then matmul
real_one_hot = F.one_hot(real_token_ids, num_classes=vocab_size).float()
real_embeds  = real_one_hot @ embedding_matrix

# Fake: softmax, then matmul (as above)
fake_probs   = F.softmax(logits / temperature, dim=-1)
fake_embeds  = fake_probs @ embedding_matrix
```

Mathematically, `one_hot @ embedding_matrix` gives the **exact same result** as
`embedding_matrix[token_id]` — selecting row 5 via one-hot multiplication is identical
to indexing row 5 directly. But now both real and fake data flow through the same
operation type. The critic can't cheat on format; it has to learn actual protein features
to distinguish real from fake.

---

## 5. The Multi-Step Problem and Straight-Through

ProtGen's generator doesn't produce a sequence in one shot. It uses **iterative
refinement**:

```
Start: [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]
                                     ↓ fill top 10% most confident
Step 1: [MASK] [MASK] [MASK] [MASK]  Ala  [MASK] [MASK] [MASK] [MASK] [MASK]
                                     ↓ fill next 10%
Step 2: [MASK] [MASK]  Gly  [MASK]  Ala  [MASK] [MASK] [MASK] [MASK] [MASK]
                                     ↓ ...repeat...
Step N:  Leu    Val    Gly    Ile    Ala    Trp    Ser    Phe    Asp    Glu
```

At each intermediate step, the generator commits specific positions — converts them
from continuous logits to hard integer tokens — and feeds them back as input to
ProtBERT for the next step. This creates **multiple gradient walls**, not just one:

```
Step 1 logits → hard token → Step 2 input → Step 2 logits → hard token → ... → final output
                   ↑ wall                                        ↑ wall
```

If we only make the **final** output soft, gradient can only reach the weights that
produced the final step's logits. The weights that decided earlier steps (which positions
to fill, which amino acids to commit) get no signal.

### Straight-through estimator

The compromise: during the **forward pass**, use hard tokens at intermediate steps (so
ProtBERT sees the same kind of input it was pre-trained on). During the **backward pass**,
pretend the hard sampling was actually soft — use the soft surrogate's gradient as an
approximation.

```
Forward:  logits → hard sample (token ID) → feed to next step
Backward: logits → (pretend it was) softmax → gradient flows through
```

This is mathematically biased (the gradient isn't exactly correct), but it's a
well-established approximation used throughout deep learning (originally from Bengio et
al., 2013). It works because the soft gradient usually points in roughly the right
direction, even if the magnitude isn't perfect.

---

## 6. Why We Truncate to K=1

Even with straight-through, backpropagating through ALL refinement steps is:

1. **Memory-expensive**: PyTorch must store the activations from every intermediate
   step to compute gradients. With 10 refinement steps and ProtBERT-sized activations,
   this can blow past GPU memory.

2. **Gradient quality degrades**: Each straight-through step introduces approximation
   error. After 10 steps of compounding approximation, the gradient signal reaching the
   earliest steps may be more noise than signal.

3. **Diminishing returns**: The final refinement step has the most direct influence on
   the output sequence. Earlier steps mostly set up context that the later steps refine.

**K=1** means: only backpropagate through the last refinement step. All earlier steps
are treated as a non-differentiable "context generator" — they produce input for the
final step, but their weights don't receive gradient from this path.

This is the same approach used by DRAKES (ICLR 2025) and DRaFT (Clark et al., ICLR
2024) for similar iterative generation architectures. Start conservative (K=1), increase
only if the adversarial signal proves too weak.

---

## 7. The KL Anchor — Preventing Reward Hacking

Once gradients flow correctly, the generator can finally optimize. But "optimize" means
"find whatever sequence fools the critic most effectively." The critic is a learned
classifier, not a physics simulation — it has blind spots.

Without an anchor, the generator can discover **adversarial sequences**: strings of amino
acids that score perfectly against the critic but are physically nonsensical — they'd
never fold into a real protein structure.

The DRAKES paper (ICLR 2025) demonstrated this directly. They fine-tuned a protein
generator with reward optimization (analogous to our adversarial critic signal). Their
ablation:

| | With KL anchor | Without KL anchor |
|---|---|---|
| Predicted stability (reward) | High | High (looks great!) |
| Actual fold fidelity (scRMSD) | 0.918 | **7.307** (garbage) |

The generator learned to hack the reward model while producing sequences that don't fold.

### How the anchor works

Add a penalty term to the generator's loss:

```python
g_loss = -critic_score + λ * KL(generator_output || frozen_pretrained_protbert_output)
```

The KL divergence measures how far the generator's probability distribution has drifted
from the original pretrained ProtBERT's distribution. The `λ` weight controls how tight
the leash is:

- **High λ**: generator stays very close to pretrained ProtBERT. Safe but slow to adapt.
- **Low λ**: generator has more freedom to explore. Faster but risk of reward hacking.
- **λ = 0**: no anchor. Free to diverge arbitrarily. This is what collapsed in DRAKES.

The frozen pretrained ProtBERT is a separate copy that doesn't get updated — it serves as
a fixed reference point. It says: "you can learn to fool the critic, but your outputs must
still look like something a protein language model would produce."

An alternative to KL divergence is a simpler **MLM cross-entropy anchor**: run the same
masked input through the frozen ProtBERT, and penalize the generator for disagreeing with
its predictions. Same idea, slightly different math.

---

## 8. The Gradient Penalty in Embedding Space

WGAN-GP (Wasserstein GAN with Gradient Penalty) stabilizes training by penalizing the
critic when its gradients become too large. The penalty is computed on **interpolated**
samples — random mixtures between real and fake data:

```python
alpha = random(0, 1)
interpolated = alpha * real_data + (1 - alpha) * fake_data
```

In the original ProtGen code, this interpolation operates on integer token IDs:

```python
# BROKEN: interpolating integers is meaningless
interpolated = alpha * 5 + (1 - alpha) * 12  # = 7.8 — what token is 7.8?
```

Token ID 7.8 doesn't exist. The current code works around this by embedding the
interpolated IDs, but the interpolation itself is in the wrong space.

With soft embeddings, both real and fake data are continuous embedding vectors.
Interpolation makes geometric sense:

```python
# CORRECT: interpolating in embedding space
interpolated = alpha * real_embeds + (1 - alpha) * fake_embeds
# This is a meaningful point in embedding space — a blend of real and fake features
```

The gradient penalty then operates on these interpolated embeddings:

```python
critic_score = critic.forward_from_embeds(interpolated)
gradients = torch.autograd.grad(critic_score, interpolated)
penalty = (gradients.norm(2) - 1) ** 2
```

This is mathematically well-defined and produces a meaningful stabilization signal.

---

## 9. Putting It All Together

### Before the fix

```
Generator (ProtBERT)
    ↓ logits
multinomial sampling
    ↓ integer token IDs ← gradient wall
Critic (ProtBERT + head)
    ↓ score
g_loss.backward()
    ↓ gradient stops at the wall
Generator weights: UNCHANGED (except weight decay erosion)
```

The GAN is not a GAN. It's a frozen sequence generator + an improving classifier.

### After the fix

```
Generator (ProtBERT)
    ↓ logits                                    ↓ logits (same)
    ↓                                           ↓
softmax(logits/T) → probs                     multinomial(probs) → hard tokens
    ↓                                           ↓
probs @ embedding_matrix → soft_embeds        → FASTA files, logging, eval metrics
    ↓ (soft path, used ONLY for critic)         (hard path, used everywhere else)
    ↓
Critic receives soft_embeds
    ↓
+ KL anchor against frozen ProtBERT
    ↓
g_loss.backward()
    ↓ gradient flows through softmax and matmul
Generator weights: UPDATED by adversarial signal
```

Two separate paths from the same logits:
- **Soft path**: for the critic during generator training. Enables gradient flow.
- **Hard path**: for everything else. Produces real sequences.

### The complete generator loss

```python
g_loss = (
    -critic(soft_embeds).mean()                          # adversarial: fool the critic
    + λ * KL(generator_probs || frozen_protbert_probs)   # anchor: don't drift from reality
)
```

### What changes in which files

| File | Change |
|------|--------|
| `models.py` | Add soft-embedding output path to generator; add `forward_from_embeds` to critic |
| `loss.py` | Rewrite `compute_gradient_penalty` to interpolate in embedding space |
| `10p_train.py` | Generator update step: use soft embeds for critic, hard tokens for everything else; add anchor loss term |
| `fully_masked_train.py` | Same changes as `10p_train.py` (or merged into single script by then) |

The evaluation pipeline (`val_metrics.py`, `generate.py`) is **unaffected** — it only
uses the hard-token path.
