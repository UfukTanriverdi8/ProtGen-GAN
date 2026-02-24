#!/usr/bin/env python3
"""
Build a master CSV from sequence text files.

- Each input .txt file: one sequence per line.
- Samples 600 sequences per file (uniform random, seed=42).
- Parses (run_name, mode, checkpoint_epoch) from filenames like:
    10p_freeze_gen_5e-6_crit_5e-5_epoch_13_full.txt
    finetuned_protbert_seeded.txt
    protbert_base_full.txt
- Outputs columns (exact order):
    id, run_name, mode, checkpoint_epoch, sequence, length, plddt, progres, max_tmscore, scaccuracy, notes
"""

import argparse
import glob
import os
import random
import re
import sys
import pandas as pd

RNG_SEED = 12
SAMPLE_PER_FILE = 10000
OUTPUT_CSV = "120k_seqs.csv"

# Your provided filenames (restrict to these to avoid accidental extra files)
EXPECTED_FILES = [
    "10p_freeze_gen_5e-6_crit_5e-5_epoch_13_full.txt",
    "10p_freeze_gen_5e-6_crit_5e-5_epoch_13_seeded.txt",
    "10p_nc8_lrgen_5e-6_lrcrit_5e-5_epoch_5_full.txt",
    "10p_nc8_lrgen_5e-6_lrcrit_5e-5_epoch_5_seeded.txt",
    "full_freeze_gen_5e-5_crit_5e-4_epoch_14_full.txt",
    "full_freeze_gen_5e-5_crit_5e-4_epoch_14_seeded.txt",
    "full_nc8_lrgen_5e-5_lrcrit_5e-4_epoch_7_full.txt",
    "full_nc8_lrgen_5e-5_lrcrit_5e-4_epoch_7_seeded.txt",
    "finetuned_protbert_full.txt",
    "finetuned_protbert_seeded.txt",
    "protbert_base_full.txt",
    "protbert_base_seeded.txt",
]

# Regex to parse filename into (run_name, checkpoint_epoch, mode)
# Matches both with and without _epoch_<int> part.
FILENAME_RE = re.compile(r"""
    ^(?P<base>.+?)          # everything up to the optional epoch part
    (?:_epoch_(?P<epoch>\d+))?  # optional epoch_X
    _(?P<mode>full|seeded)  # mode suffix
    \.txt$
""", re.VERBOSE)

def parse_filename(fname: str):
    """
    Returns (run_name, mode, checkpoint_epoch_str_or_empty)
    For baselines without epoch, checkpoint_epoch -> "" (empty)
    """
    bn = os.path.basename(fname)
    m = FILENAME_RE.match(bn)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {bn}")
    base = m.group("base")
    epoch = m.group("epoch")
    mode = m.group("mode")

    # run_name is the base without the trailing _epoch_* and _mode.
    # For files with epoch, base already excludes it by regex design.
    # Examples:
    #   10p_freeze_gen_5e-6_crit_5e-5  (epoch=13)
    #   finetuned_protbert             (epoch=None)
    run_name = base

    checkpoint_epoch = epoch if epoch is not None else ""

    return run_name, mode, checkpoint_epoch

def read_sequences(path: str):
    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().upper()
            if not s:
                continue
            seqs.append(s)
    return seqs

def main():
    parser = argparse.ArgumentParser(description="Build master CSV from sequence files.")
    parser.add_argument(
        "--dir", default=".", help="Directory containing the .txt files (default: current dir)"
    )
    parser.add_argument(
        "--out", default=OUTPUT_CSV, help=f"Output CSV filename (default: {OUTPUT_CSV})"
    )
    parser.add_argument(
        "--n", type=int, default=SAMPLE_PER_FILE,
        help=f"Sample size per file (default: {SAMPLE_PER_FILE})"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="If set, fail if any EXPECTED_FILES are missing. Otherwise, only use those that exist."
    )
    args = parser.parse_args()

    random.seed(RNG_SEED)

    # Resolve files
    files_present = []
    for fn in EXPECTED_FILES:
        full = os.path.join(args.dir, fn)
        if os.path.isfile(full):
            files_present.append(full)
        elif args.strict:
            print(f"ERROR: Missing expected file: {full}", file=sys.stderr)
            sys.exit(1)

    if not files_present:
        print("No expected files found. Check --dir or file names.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for path in sorted(files_present):
        run_name, mode, checkpoint_epoch = parse_filename(path)
        seqs = read_sequences(path)
        if not seqs:
            continue

        k = min(args.n, len(seqs))
        # Sample uniformly at random (reproducible due to fixed seed)
        sample_indices = sorted(random.sample(range(len(seqs)), k))

        # id resets per (run_name, mode, checkpoint_epoch)
        for i, idx in enumerate(sample_indices, start=1):
            seq = seqs[idx]
            rows.append({
                "id": i,  # 1..N within this file/case
                "run_name": run_name,
                "mode": mode,
                "checkpoint_epoch": checkpoint_epoch,  # empty string if N/A
                "sequence": seq,
                "length": len(seq),
                "plddt": "",         # to be filled later
                "progres": "",       # to be filled later
                "max_tmscore": "",   # to be filled later
                "seq_identity": "",
                "seq_similarity": "",
                "sc_accuracy": "",    # to be filled later
                "notes": "",         # optional flags later
            })

    # Build DataFrame with exact column order
    columns = [
        "id",
        "run_name",
        "mode",
        "checkpoint_epoch",
        "sequence",
        "length",
        "plddt",
        "progres",
        "max_tmscore",
        "scaccuracy",
        "notes",
    ]
    df = pd.DataFrame(rows, columns=columns)

    # Sort (optional but nice): by run_name, mode, checkpoint, then id
    df["checkpoint_epoch_sort"] = df["checkpoint_epoch"].replace("", "-1").astype(int)
    df = df.sort_values(by=["run_name", "mode", "checkpoint_epoch_sort", "id"]).drop(columns=["checkpoint_epoch_sort"])

    # Write CSV (UTF-8)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")

if __name__ == "__main__":
    main()

