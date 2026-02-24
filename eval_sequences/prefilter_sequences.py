# Cell 1 — setup
import pandas as pd
import numpy as np

INPUT_CSV  = "120k_eval/120k_eval_seqs.csv"      # change path as needed
OUTPUT_CSV = "120k_eval_seqs_prefiltered.csv" # change path as needed

MAX_LEN = 350   # keep consistent with your evaluation script
DROP_X = True   # skip any sequence containing 'X'
# Cell 2 — load + detect sequence column
df = pd.read_csv(INPUT_CSV)
candidates = ["sequence", "seq"]
seq_col = next((c for c in candidates if c in df.columns), None)
if seq_col is None:
    raise ValueError(f"Could not find sequence column. Tried: {candidates}. Columns: {list(df.columns)}")

print("Sequence column:", seq_col)
print("Rows:", len(df))
# Cell 3 — drop max TM column(s) if present
tm_cols = ["max_tmscore", "max_tm", "max_tm_score"]
drop_these = [c for c in tm_cols if c in df.columns]
if drop_these:
    df = df.drop(columns=drop_these)
print("Dropped:", drop_these)
# Cell 4 — basic cleaning + filters
seqs = df[seq_col].astype(str).str.strip()

# Treat empty / "nan" strings as missing
is_missing = seqs.eq("") | seqs.str.lower().eq("nan")

# Length filter
lengths = seqs.str.len()
too_long = lengths > MAX_LEN

# X filter
has_x = seqs.str.contains("X") if DROP_X else pd.Series(False, index=df.index)

before = len(df)
# --- Separated drop logic ---
mask_missing  = is_missing
mask_too_long = ~is_missing & too_long
mask_has_x    = ~is_missing & ~too_long & has_x   # not too long but has X
mask_keep     = ~is_missing & ~too_long & ~has_x

df_missing  = df.loc[mask_missing]
df_too_long = df.loc[mask_too_long]
df_has_x    = df.loc[mask_has_x]      # <-- sequences dropped specifically because of X
df_filt     = df.loc[mask_keep].copy()
df_filt["length"] = df_filt[seq_col].astype(str).str.len()

print(f"Before        : {before}")
print(f"Dropped missing  : {len(df_missing)}")
print(f"Dropped too long : {len(df_too_long)}")
print(f"Dropped has X    : {len(df_has_x)}")
print(f"Kept             : {len(df_filt)}")

# Cell 5 — X amount distribution among dropped-because-of-X sequences
print("\nX count distribution in sequences dropped due to X:")
x_counts = df_has_x[seq_col].str.count("X")
print(x_counts.value_counts().sort_index())



# df_filt.to_csv(OUTPUT_CSV)

# print("X distribution:")
# print(df_filt[seq_col].str.count("X").value_counts())