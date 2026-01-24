#!/usr/bin/env python3
# generate.py
import os
import math
import random
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from models import Generator

# ---- fixed paths on MN5 ----
MAIN_FOLDER = "/gpfs/projects/etur29/ufuk/"
CKPT_ROOT = os.path.join(MAIN_FOLDER, "gan-checkpoints")
PROTBERT_FINETUNED = os.path.join(MAIN_FOLDER, "dynamic-finetuned-protbert")
PROTBERT_BASE = os.path.join(MAIN_FOLDER, "protbert-base")  

# ---------------- utils ----------------
def load_length_distribution(dataset_path: str, max_len: int, min_len: int = 30) -> List[int]:
    lengths = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            seq = line.strip().replace(" ", "")
            if not seq:
                continue
            L = len(seq)
            if min_len <= L <= max_len:
                lengths.append(L)
    if not lengths:
        raise ValueError(f"No sequence lengths found in range [{min_len}, {max_len}] from {dataset_path}.")
    return lengths

def load_seed_sequences(seed_file: str, max_len: int, min_len: int = 30) -> List[str]:
    seqs = []
    with open(seed_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().replace(" ", "")
            if not s:
                continue
            L = len(s)
            if min_len <= L <= max_len:
                seqs.append(s[:max_len])
    if not seqs:
        raise ValueError(f"No usable seed sequences in range [{min_len}, {max_len}] from {seed_file}.")
    return seqs

def sample_sequence_length(lengths: List[int], rng: random.Random) -> int:
    return rng.choice(lengths)

def to_token_ids(seq: str, tokenizer) -> List[int]:
    # ProtBERT expects space-separated amino acids
    return tokenizer.encode(" ".join(list(seq)), add_special_tokens=False)

def safe_local_from_pretrained(path: str, device: str):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Model dir not found: {path}")
    if not os.path.exists(os.path.join(path, "config.json")):
        raise FileNotFoundError(f"Missing config.json in {path} (did save_pretrained run?)")
    model = AutoModelForMaskedLM.from_pretrained(path, local_files_only=True)
    return model.to(device)

# -------------- path logic --------------
def resolve_model_path(ckpt_id: str) -> Tuple[str, str]:
    """
    Returns (model_path, tag)
    tag in {"trained", "finetuned", "base"} for logging.
    """
    ckpt_id = ckpt_id.strip().strip("/")
    specials_finetuned = {"finetuned_protbert", "finetuned-protbert"}   # clean single alias
    specials_base = {"protbert_base", "protbert-base"}             # keep one alias for base

    if ckpt_id in specials_finetuned:
        return PROTBERT_FINETUNED, "finetuned"
    if ckpt_id in specials_base:
        return PROTBERT_BASE, "base"

    # trained GAN generator
    return os.path.join(CKPT_ROOT, ckpt_id, "generator_bert"), "trained"

def derive_out_prefix(ckpt_id: str) -> str:
    ckpt_id = ckpt_id.strip().strip("/")
    specials = {"finetuned-protbert", "finetuned_protbert", "protbert_base", "protbert-base", "base_protbert"}
    if ckpt_id in specials:
        return ckpt_id
    parts = ckpt_id.split("/")
    if len(parts) == 2:
        run_name, epoch_tag = parts
        return f"{run_name}_{epoch_tag}"
    return ckpt_id.replace("/", "_")

# -------------- builders --------------
def build_fully_masked_batch(batch_sizes: List[int], cls_id: int, sep_id: int, mask_id: int, pad_id: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    max_L = max(batch_sizes)
    B = len(batch_sizes)
    input_ids = torch.full((B, max_L + 2), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros_like(input_ids, dtype=torch.long, device=device)
    for i, L in enumerate(batch_sizes):
        attn[i, : L + 2] = 1
        input_ids[i, 0] = cls_id
        input_ids[i, L + 1] = sep_id
        if L > 0:
            input_ids[i, 1 : L + 1] = mask_id
    return input_ids, attn

def build_seeded_batch(seqs: List[str], tokenizer, keep_init: float, cls_id: int, sep_id: int, mask_id: int, pad_id: int, device, rng: random.Random) -> Tuple[torch.Tensor, torch.Tensor]:
    # tokenize each seed (no specials), then place kept tokens into the masked template
    tok_lists = [to_token_ids(s, tokenizer) for s in seqs]
    Ls = [len(toks) for toks in tok_lists]
    max_L = max(Ls)
    B = len(seqs)

    input_ids = torch.full((B, max_L + 2), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros_like(input_ids, dtype=torch.long, device=device)

    for i, toks in enumerate(tok_lists):
        L = len(toks)
        attn[i, : L + 2] = 1
        input_ids[i, 0] = cls_id
        input_ids[i, L + 1] = sep_id
        if L > 0:
            input_ids[i, 1 : L + 1] = mask_id
            # choose positions to KEEP (≈10% of L)
            k = max(1, int(round(keep_init * L)))
            keep_positions = rng.sample(range(1, L + 1), k)  # positions in [1..L]
            for pos in keep_positions:
                input_ids[i, pos] = toks[pos - 1]
    return input_ids, attn

# -------------- generators --------------
@torch.no_grad()
def generate_full_mode_sequences(generator: Generator, tokenizer, num_sequences: int, length_pool: List[int],
                                 keep_percent: float = 0.10, temperature: float = 1.0, batch_size: int = 64,
                                 device: str = "cuda", rng_seed: int = 42) -> List[str]:
    assert 0.0 < keep_percent <= 1.0
    device = torch.device(device)
    generator.eval()

    rng = random.Random(rng_seed)
    mask_id, cls_id, sep_id, pad_id = tokenizer.mask_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has no pad_token_id; please set one.")

    out, n_done = [], 0
    max_steps = math.ceil(1.0 / keep_percent)

    while n_done < num_sequences:
        this_batch = min(batch_size, num_sequences - n_done)
        batch_lengths = [sample_sequence_length(length_pool, rng) for _ in range(this_batch)]
        input_ids, attn = build_fully_masked_batch(batch_lengths, cls_id, sep_id, mask_id, pad_id, device)
        current_rate = 1.0
        for _ in range(max_steps):
            if (input_ids == mask_id).sum().item() == 0:
                break
            input_ids = generator.generate(
                input_ids, attention_mask=attn,
                keep_percent=keep_percent, current_rate=current_rate, temperature=temperature,
            )
            current_rate = max(0.0, current_rate - keep_percent)
        for i in range(this_batch):
            out.append(tokenizer.decode(input_ids[i], skip_special_tokens=True).replace(" ", ""))
        n_done += this_batch
    return out

@torch.no_grad()
def generate_seeded_mode_sequences(generator: Generator, tokenizer, num_sequences: int, seed_pool: List[str],
                                   keep_init: float = 0.10, keep_percent: float = 0.10, temperature: float = 1.0,
                                   batch_size: int = 64, device: str = "cuda", rng_seed: int = 123) -> List[str]:
    """
    Seeded-10%: keep_init (~10%) real tokens, mask the rest, then iteratively fill keep_percent per step.
    """
    assert 0.0 < keep_init < 1.0 and 0.0 < keep_percent <= 1.0
    device = torch.device(device)
    generator.eval()

    rng = random.Random(rng_seed)
    mask_id, cls_id, sep_id, pad_id = tokenizer.mask_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has no pad_token_id; please set one.")

    out, n_done = [], 0
    max_steps = max(1, math.ceil((1.0 - keep_init) / keep_percent)) + 1  # generous bound
    N = len(seed_pool)

    while n_done < num_sequences:
        this_batch = min(batch_size, num_sequences - n_done)
        # sample seeds with replacement
        batch_seqs = [seed_pool[rng.randrange(N)] for _ in range(this_batch)]
        input_ids, attn = build_seeded_batch(batch_seqs, tokenizer, keep_init, cls_id, sep_id, mask_id, pad_id, device, rng)
        current_rate = 1.0 - keep_init
        for _ in range(max_steps):
            if (input_ids == mask_id).sum().item() == 0:
                break
            input_ids = generator.generate(
                input_ids, attention_mask=attn,
                keep_percent=keep_percent, current_rate=current_rate, temperature=temperature,
            )
            current_rate = max(0.0, current_rate - keep_percent)
        for i in range(this_batch):
            out.append(tokenizer.decode(input_ids[i], skip_special_tokens=True).replace(" ", ""))
        n_done += this_batch
    return out

# -------------- main --------------
def main():
    random_seed = random.randint(1, 10000)
    ap = argparse.ArgumentParser(description="Generate sequences in full & seeded modes.")
    ap.add_argument("--ckpt_id", required=True, help="Trained: 'run/epoch_XX' | Finetuned: '10p_untrained'/'full_untrained' | Base: 'protbert_base'")
    ap.add_argument("--dataset_path", type=str, default="./data/dnmt_unformatted.txt", help="For length sampling (full mode).")
    ap.add_argument("--seed_file", type=str, default=None, help="Held-out seeds file (one seq per line). If None, uses dataset_path.")
    ap.add_argument("--num_full", type=int, default=5000, help="How many full-mode sequences to generate.")
    ap.add_argument("--num_seeded", type=int, default=5000, help="How many seeded-10% sequences to generate.")
    ap.add_argument("--keep_init", type=float, default=0.10, help="Initial kept fraction for seeded mode.")
    ap.add_argument("--keep_percent", type=float, default=0.10, help="Per-iteration fill fraction.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_len", type=int, default=600)
    ap.add_argument("--min_len", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--rng_seed", type=int, default=random_seed)
    ap.add_argument("--tokenizer_dir", type=str, default=PROTBERT_FINETUNED, help="Tokenizer path (defaults to finetuned).")
    ap.add_argument("--out_dir", type=str, default="./results/")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_prefix = derive_out_prefix(args.ckpt_id)
    full_out = str(Path(args.out_dir) / f"{out_prefix}_full.txt")
    seeded_out = str(Path(args.out_dir) / f"{out_prefix}_seeded.txt")

    # Resolve model and tokenizer
    model_path, tag = resolve_model_path(args.ckpt_id)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, do_lower_case=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    protbert = safe_local_from_pretrained(model_path, args.device)
    if tokenizer.pad_token_id >= protbert.get_input_embeddings().weight.size(0):
        protbert.resize_token_embeddings(len(tokenizer))
    generator = Generator(protbert_model=protbert).to(args.device)
    generator.eval()

    # Full-mode lengths
    lengths = load_length_distribution(args.dataset_path, max_len=args.max_len, min_len=args.min_len)

    # Seeded seeds
    seed_source = args.seed_file if args.seed_file is not None else args.dataset_path
    seeds = load_seed_sequences(seed_source, max_len=args.max_len, min_len=args.min_len)

    # FULL
    seqs_full = []
    if args.num_full > 0:
        seqs_full = generate_full_mode_sequences(
            generator=generator, tokenizer=tokenizer, num_sequences=args.num_full,
            length_pool=lengths, keep_percent=args.keep_percent, temperature=args.temperature,
            batch_size=args.batch_size, device=args.device, rng_seed=args.rng_seed,
        )
        with open(full_out, "w", encoding="utf-8") as f:
            for s in seqs_full: f.write(s + "\n")
        print(f"[{tag}] Wrote {len(seqs_full)} full-mode sequences → {full_out}")

    # SEEDED (10%)
    seqs_seeded = []
    if args.num_seeded > 0:
        seqs_seeded = generate_seeded_mode_sequences(
            generator=generator, tokenizer=tokenizer, num_sequences=args.num_seeded,
            seed_pool=seeds, keep_init=args.keep_init, keep_percent=args.keep_percent,
            temperature=args.temperature, batch_size=args.batch_size, device=args.device,
            rng_seed=args.rng_seed + 1337,
        )
        with open(seeded_out, "w", encoding="utf-8") as f:
            for s in seqs_seeded: f.write(s + "\n")
        print(f"[{tag}] Wrote {len(seqs_seeded)} seeded-10% sequences → {seeded_out}")

    print(f"[DONE] {out_prefix} | full={len(seqs_full)} | seeded={len(seqs_seeded)}")

if __name__ == "__main__":
    main()

