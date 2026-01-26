# Cell 1 — setup
import pandas as pd
import numpy as np

INPUT_CSV  = "eval_seqs_23_01_25.csv"      # change path as needed
OUTPUT_CSV = "eval_seqs_23_01_25_prefiltered.csv" # change path as needed

MAX_LEN = 400   # keep consistent with your evaluation script
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
df_filt = df.loc[~is_missing & ~too_long & ~has_x].copy()
df_filt["length"] = df_filt[seq_col].astype(str).str.len()

print("Before:", before)
print("After :", len(df_filt))
print("Dropped missing:", int(is_missing.sum()))
print("Dropped too long:", int(too_long.sum()))
print("Dropped X:", int(has_x.sum()))
# Cell 5 — length distribution (basic)
print(df_filt["length"].describe())


df_filt.to_csv(OUTPUT_CSV)