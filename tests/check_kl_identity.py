"""
KL identity sanity check for compute_kl_anchor (models.py).

Mathematical fact being tested: KL(P || P) = 0 for any distribution P. If the
generator and the reference model have identical weights and see the identical
input, compute_kl_anchor MUST return ~0. This is independent of whatever the
training curves show — it verifies the plumbing (right tensors passed to the
right arguments, softmax actually applied, positions correctly aligned), not
just that the numbers look "plausible" during training.

A negative control is included: same setup but with the reference weights
perturbed, where KL should be clearly nonzero. Without this, a KL that's
always ~0 (e.g. due to remask_positions accidentally empty, or ref_protbert
not actually being used) would look like a "pass" for the wrong reason.

Run (from repo root): conda run -n protgen-gan python3 tests/check_kl_identity.py
"""

import copy
import sys
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROTBERT_PATH
from models import Critic, Generator, compute_kl_anchor, compute_soft_embeds

torch.manual_seed(0)
device = "cuda"  # CUDA_VISIBLE_DEVICES pins this to a specific GPU at invocation

tokenizer = AutoTokenizer.from_pretrained(PROTBERT_PATH, do_lower_case=False)

# Toy batch — any real amino-acid sequences work, the identity holds regardless.
seqs = [
    "M K T A Y I A K Q R Q I S F V K S H F S R Q L E E R L G L I E V Q A",
    "M A L W M R L L P L L A L L A L W G P D P A A A F V N Q H L C G S H",
]
enc = tokenizer(seqs, padding=True, return_tensors="pt")
input_ids = enc["input_ids"].to(device)
attn_mask = enc["attention_mask"].float().to(device)

# Generator and reference: two SEPARATE loads of the SAME checkpoint. Separate
# objects on purpose (not the same Python object) so the test can't pass just
# because generator and ref_protbert alias one another.
generator_protbert = AutoModelForMaskedLM.from_pretrained(PROTBERT_PATH).to(device)
ref_protbert = AutoModelForMaskedLM.from_pretrained(PROTBERT_PATH).to(device)
generator_protbert.eval()
ref_protbert.eval()
for p in ref_protbert.parameters():
    p.requires_grad = False

generator = Generator(
    protbert_model=generator_protbert, mask_token_id=tokenizer.mask_token_id
).to(device)
generator.eval()

# Critic is only needed here to supply an embedding table to compute_soft_embeds;
# its classifier head is never touched by this test.
critic = Critic(protbert_model=copy.deepcopy(generator_protbert)).to(device)
critic.eval()


def run_kl(ref_model, label):
    with torch.no_grad():
        _soft_embeds, gen_probs, temperature, remask_positions, masked_input = (
            compute_soft_embeds(
                generator,
                critic,
                input_ids,
                attn_mask,
                min_temp=1.0,
                max_temp=1.0,
                tokenizer=tokenizer,
                remask_frac=0.5,
            )
        )
        kl = compute_kl_anchor(
            gen_probs, ref_model, masked_input, attn_mask, temperature, remask_positions
        )
    n_remasked = remask_positions.sum().item()
    print(f"[{label}] n_remasked_positions={n_remasked}  kl_loss={kl.item():.8f}")
    return kl.item()


print("=== Identity case: generator weights == reference weights ===")
kl_identity = run_kl(ref_protbert, "identical weights")
assert kl_identity < 1e-3, (
    f"FAIL: identical generator/reference gave KL={kl_identity}, expected ~0. "
    "Something is feeding compute_kl_anchor mismatched inputs, or a distribution "
    "isn't actually being softmax'd before comparison."
)
print("PASS: KL(gen || ref) ~= 0 when gen and ref are the same weights.\n")

print("=== Negative control: reference weights perturbed ===")
ref_protbert_perturbed = AutoModelForMaskedLM.from_pretrained(PROTBERT_PATH).to(device)
ref_protbert_perturbed.eval()
with torch.no_grad():
    for p in ref_protbert_perturbed.parameters():
        p.add_(torch.randn_like(p) * 0.1)
        p.requires_grad = False
kl_perturbed = run_kl(ref_protbert_perturbed, "perturbed weights")
assert kl_perturbed > 1e-2, (
    f"FAIL: perturbed reference gave KL={kl_perturbed}, expected clearly nonzero. "
    "If this is also ~0, compute_kl_anchor may not actually be using ref_protbert's "
    "output (e.g. remask_positions is empty, or the wrong tensor is being compared)."
)
print("PASS: KL(gen || ref) is clearly nonzero when weights differ.\n")

print(
    "compute_kl_anchor identity check: PASSED both the identity and negative-control cases."
)
