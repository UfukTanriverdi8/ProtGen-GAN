#!/usr/bin/env python3
# run_mass_eval.py

import os
import sys
import argparse
import time
import math
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib

# --- Import metric utilities from your val_metrics.py ---
import progres as pg
from val_metrics import (
    calculate_plddt_scores_and_save_pdb,
    get_mpnn_sequence_from_pdb,
    compute_alignment_identity_and_similarity,
)

import torch
from transformers import AutoTokenizer, EsmForProteinFolding
torch.backends.cuda.matmul.allow_tf32 = True

#from transformers.models.esm import EsmForProteinFolding

# =======================
# User-editable constants
# =======================
TARGET_DNMT_SEQUENCE = "IRVLSLFDGIATGLLVLKDLGIQVDRYIASEVCEDSITVGMVRHQGKIMYVGDVRSVTQKHIQEWGPFDLVIGGSPCNDLSIVNPARKGLYEGTGRLFFEFYRLLHDARPKEGDDRPFFWLFENVVAMGVSDKRDISRFLESNPVMIDAKEVSAAHRARYFWGNLPGMNRPLASTVNDKLELQECLEHGRIAKFSKVRTITTRSNSIKQGKDQHFPVFMNEKEDILWCTEMERVFGFPVHYTDVSNMSRLARQRLLGRSWSVPVIRHLFAPLKEYFACV"  # <-- fill this with your reference DNMT sequence (string)
REFERENCE_PDB = "validation/pdbs/reference/DNMT3A.pdb"  # path to your reference PDB
OUTPUT_CSV = "eval_results/eval_seqs_23_01_25_filled.csv"
DEFAULT_RUN_NAME = "mass_eval"
DEFAULT_BATCH_SIZE = 8
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Column names (robust to missing; created if needed)
METRIC_COLUMNS = [
    "uid",
    "plddt",
    "progres",
    "scaccuracy",
    "seq_id",
    "seq_similarity",
    "pdb_path",
]

# =======================
# Helper functions
# =======================

def find_sequence_column(df: pd.DataFrame) -> str:
    """Best-effort detection of the sequence column."""
    candidates = ["sequence", "seq", "generated_sequence", "protein"]
    for c in candidates:
        if c in df.columns:
            return c
    # If not found, raise a clear error
    raise ValueError(
        f"Could not find a sequence column. Tried {candidates}. "
        f"Please ensure your CSV has one of these column names."
    )

def ensure_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add metric columns if missing."""
    for col in METRIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df

def ensure_uid_column(df: pd.DataFrame) -> pd.DataFrame:
    if "uid" in df.columns and df["uid"].notna().any():
        return df

    has_run = "run_name" in df.columns
    has_id = "id" in df.columns

    if not has_run:
        # Fall back: stable hash of row index + sequence length if possible
        seq_col = next((c for c in ["sequence","seq","generated_sequence","protein"] if c in df.columns), None)
        lengths = df[seq_col].astype(str).str.len() if seq_col else pd.Series(0, index=df.index)
        base = ("idx:" + df.index.astype(str) + "|len:" + lengths.astype(str)).astype(str)
    else:
        # Use run_name + id if available; else run_name + row index
        seq_col = next((c for c in ["sequence","seq","generated_sequence","protein"] if c in df.columns), None)
        lengths = df[seq_col].astype(str).str.len() if seq_col else pd.Series(0, index=df.index)

        if has_id:
            base = df["run_name"].astype(str) + "|" + df["id"].astype(str) + "|" + lengths.astype(str)
        else:
            base = df["run_name"].astype(str) + "|" + df.index.astype(str) + "|" + lengths.astype(str)

    # 10-digit numeric uid from blake2b (low collision risk for your scale)
    def to_uid(s: str) -> int:
        h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()  # 64-bit
        n = int.from_bytes(h, "big")
        return n % 10**10  # keep 12 digits

    df["uid"] = base.map(to_uid).astype("int64")
    return df


def is_row_filled(row) -> bool:
    """Decide if a row's metrics are already filled (skip it)."""
    required = ["plddt", "progres", "scaccuracy", "seq_id", "seq_similarity"]
    return all(pd.notna(row.get(col, np.nan)) for col in required)

def save_checkpoint(df: pd.DataFrame, out_path: str):
    tmp_path = out_path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)

def load_or_init_csv(input_csv: str, output_csv: str) -> pd.DataFrame:
    """If OUTPUT_CSV exists, resume from it. Otherwise start from input and create OUTPUT_CSV."""
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        # Make sure any new columns are present
        df = ensure_metric_columns(df)
        df = ensure_uid_column(df)
        return df
    else:
        df = pd.read_csv(input_csv)
        df = ensure_metric_columns(df)
        save_checkpoint(df, output_csv)
        return df

def batch_indices(n_rows: int, batch_size: int):
    for start in range(0, n_rows, batch_size):
        yield start, min(start + batch_size, n_rows)

def prepare_batch_sequences(df: pd.DataFrame, seq_col: str, idxs: list):
    seqs = []
    row_ids = []
    for i in idxs:
        row = df.iloc[i]
        if is_row_filled(row):
            continue
        s = str(row[seq_col]).strip()
        if not s or s.lower() == "nan":
            continue
        seqs.append(s)
        row_ids.append(i)
    return seqs, row_ids

def make_run_folder(base_run_name: str, batch_no: int) -> str:
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_tag = f"_a{int(task_id):04d}" if task_id is not None else ""
    folder = f"./validation/pdbs/{base_run_name}{task_tag}_b{batch_no:05d}"
    os.makedirs(folder, exist_ok=True)
    return folder

def pdb_path_for_index(run_folder: str, local_idx: int) -> str:
    return os.path.join(run_folder, f"generated_protein_{local_idx}.pdb")



# =======================
# Main evaluation routine
# =======================

def main():
    parser = argparse.ArgumentParser(description="Mass evaluation with checkpointing & batching.")
    parser.add_argument("--csv", required=True, help="Path to input sequences CSV (e.g., eval_seqs_filtered.csv)")
    parser.add_argument("--out", default=OUTPUT_CSV, help=f"Output CSV (default: {OUTPUT_CSV})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size (default: 8)")
    parser.add_argument("--run_name", default=DEFAULT_RUN_NAME, help="Base run name for PDB folders")
    parser.add_argument("--device", default=DEVICE, help=f"Device (default: {DEVICE})")
    parser.add_argument("--start", type=int, default=0, help="Start row (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End row (exclusive)")
    args = parser.parse_args()

    if not TARGET_DNMT_SEQUENCE:
        print("WARNING: TARGET_DNMT_SEQUENCE is empty. seq_id and seq_similarity will be NaN.", file=sys.stderr)

    # Load / resume CSV
    df = load_or_init_csv(args.csv, args.out)
    seq_col = find_sequence_column(df)

    total_rows = len(df)
    start_row = max(0, args.start)
    end_row = min(total_rows, args.end) if args.end is not None else total_rows


    # Load ESMFold once
    print("Loading ESMFold model & tokenizer...")
    MAIN_FOLDER = "/gpfs/projects/etur29/ufuk/"
    esmfold_path = os.path.join(MAIN_FOLDER, "esmfold")
    esmfold_model = EsmForProteinFolding.from_pretrained(esmfold_path, low_cpu_mem_usage=True).to(args.device)
    esmfold_tokenizer = AutoTokenizer.from_pretrained(esmfold_path)
    esmfold_model.esm = esmfold_model.esm.half()
    esmfold_model.eval()
    torch.set_grad_enabled(False)

    total_rows = len(df)
    batch_no = 0

    for start, end in batch_indices(end_row - start_row, args.batch_size):
        start += start_row
        end += start_row
        batch_no += 1
        idxs = list(range(start, end))
        # Prepare this batch: skip already-filled rows
        seqs, row_ids = prepare_batch_sequences(df, seq_col, idxs)
        if not row_ids:
            # Nothing to do in this segment
            continue

        print(f"\n=== Batch {batch_no} | rows {start}-{end-1} | evaluating {len(row_ids)} sequences ===")

        # Create a unique run folder for this batch (so PDB names start at 0 each batch safely)
        run_folder = make_run_folder(args.run_name, batch_no)

        # ---- 1) ESMFold: pLDDT + save PDBs (batched inside the function) ----
        # This function expects a run_name (folder suffix) and names as generated_protein_{i}.pdb
        # We'll pass a run_name unique per batch (based on batch_no)
        _avg_plddt, plddt_scores = calculate_plddt_scores_and_save_pdb(
            generated_sequences=seqs,
            folding_tokenizer=esmfold_tokenizer,
            folding_model=esmfold_model,
            num_sequences=len(seqs),
            batch_size=min(args.batch_size, 8),  # small sub-batches to be safe
            device=args.device,
            run_name=os.path.basename(run_folder) 
        )
        # We still need per-sequence pLDDT values; the function returns only an average.
        # As a practical workaround, we’ll compute per-sequence average pLDDT from saved PDB later if needed.
        # For now, we store the batch average into each row (common pragmatic choice).
        # If you want exact per-sequence pLDDT, we can extend val_metrics to return the list.

        # PDB files for this batch (in order)
        pdb_files = [pdb_path_for_index(run_folder, i) for i in range(len(seqs))]

        # ---- 2) Progres per sequence (sequential) ----
        progres_scores = []
        for p in pdb_files:
            try:
                score = pg.progres_score(REFERENCE_PDB, p)
            except Exception as e:
                print(f"Progres error for {p}: {e}", file=sys.stderr)
                score = np.nan
            progres_scores.append(score)

        # ---- 3) Foldseek Max TM per sequence ----
        #max_tms = compute_foldseek_max_tm_for_batch(pdb_files)
        #clean_m8_folder()  # Clean up downloaded m8 files to save space

        # ---- 4) ProteinMPNN → scAccuracy (identity vs generated) per sequence ----
        scaccs = []
        for local_idx, (pdb_file, gen_seq) in enumerate(zip(pdb_files, seqs)):
            if not os.path.exists(pdb_file):
                print(f"PDB missing for scAccuracy: {pdb_file}", file=sys.stderr)
                scaccs.append(np.nan)
                continue
            try:
                pred_seq = get_mpnn_sequence_from_pdb(pdb_file, device=args.device)
                iden, sim, _, _ = compute_alignment_identity_and_similarity(gen_seq, pred_seq)
                scaccs.append(iden)  # scAccuracy = identity (as defined)
            except Exception as e:
                print(f"MPNN/scAccuracy error for {pdb_file}: {e}", file=sys.stderr)
                scaccs.append(np.nan)

        # ---- 5) seq_id & seq_similarity vs DNMT reference (PairwiseAligner) ----
        seq_ids, seq_sims = [], []
        for gen_seq in seqs:
            if not TARGET_DNMT_SEQUENCE:
                seq_ids.append(np.nan)
                seq_sims.append(np.nan)
                continue
            try:
                iden, sim, _, _ = compute_alignment_identity_and_similarity(gen_seq, TARGET_DNMT_SEQUENCE)
                seq_ids.append(iden)
                seq_sims.append(sim)
            except Exception as e:
                print(f"DNMT alignment error: {e}", file=sys.stderr)
                seq_ids.append(np.nan)
                seq_sims.append(np.nan)

        # ---- 6) Update CSV rows for this batch ----
        for j, row_idx in enumerate(row_ids):
            df.at[row_idx, "plddt"] = float(plddt_scores[j]) if j < len(plddt_scores) else np.nan
            df.at[row_idx, "progres"] = float(progres_scores[j]) if not math.isnan(progres_scores[j]) else np.nan
            #df.at[row_idx, "max_tm"] = float(max_tms[j]) if not math.isnan(max_tms[j]) else np.nan
            df.at[row_idx, "scaccuracy"] = float(scaccs[j]) if not math.isnan(scaccs[j]) else np.nan
            df.at[row_idx, "seq_id"] = float(seq_ids[j]) if not math.isnan(seq_ids[j]) else np.nan
            df.at[row_idx, "seq_similarity"] = float(seq_sims[j]) if not math.isnan(seq_sims[j]) else np.nan
            df.at[row_idx, "pdb_path"] = pdb_files[j]

        # ---- 7) Save checkpoint after every batch ----
        save_checkpoint(df, args.out)
        print(f"Saved checkpoint {args.out}")

    print("\nAll done ")

if __name__ == "__main__":
    main()
