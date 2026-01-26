import glob
import pandas as pd
import numpy as np

# Adjust these paths/patterns to your actual filenames
PARTS_GLOB = "eval_results/masseval_120k_part_*.csv"
OUT_FINAL  = "eval_results/masseval_120k_final.csv"

files = sorted(glob.glob(PARTS_GLOB))
if not files:
    raise FileNotFoundError(f"No files matched: {PARTS_GLOB}")

dfs = []
for f in files:
    df = pd.read_csv(f)
    df["__source_file"] = f
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

# Choose merge key
if "uid" in all_df.columns:
    key_cols = ["uid"]
elif "run_name" in all_df.columns and "id" in all_df.columns:
    key_cols = ["run_name", "id"]
else:
    raise ValueError("No suitable key found. Need 'uid' or ('run_name','id').")

# For each column, take first non-null within each group
def first_non_null(s: pd.Series):
    s2 = s.dropna()
    return s2.iloc[0] if len(s2) else np.nan

merged = all_df.groupby(key_cols, as_index=False).agg(first_non_null)

# Optional: if you have a sequence column, keep a length column for sanity checks
seq_candidates = ["sequence", "seq", "generated_sequence", "protein"]
seq_col = next((c for c in seq_candidates if c in merged.columns), None)
if seq_col:
    merged["seq_len"] = merged[seq_col].astype(str).str.len()

merged.to_csv(OUT_FINAL, index=False)
print("Merged parts:", len(files))
print("Rows in merged:", len(merged))
print("Saved:", OUT_FINAL)

# Quick sanity: how many rows still missing key metrics
for col in ["plddt", "progres", "scaccuracy", "seq_id", "seq_similarity"]:
    if col in merged.columns:
        print(col, "missing:", int(merged[col].isna().sum()))
