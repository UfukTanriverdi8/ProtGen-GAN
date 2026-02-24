import glob
import pandas as pd
import numpy as np

eval_id = "120k"

files = sorted(glob.glob(f"{eval_id}_eval/masseval_{eval_id}_part_*.csv"))
dfs = [pd.read_csv(f, low_memory=False) for f in files]
all_df = pd.concat(dfs, ignore_index=True)

# Make sure uid exists (fallback if needed)
if "uid" not in all_df.columns or all_df["uid"].notna().sum() == 0:
    all_df["uid"] = all_df["run_name"].astype(str) + ":" + all_df["id"].astype(str)

def first_non_null(s):
    s2 = s.dropna()
    return s2.iloc[0] if len(s2) else np.nan

merged = all_df.groupby("uid", as_index=False).agg(first_non_null)

print("Final rows:", len(merged))
print("Non-null plddt:", merged["plddt"].notna().sum())

merged.to_csv(f"{eval_id}_eval/{eval_id}_eval_seqs_final.csv", index=False)
print("Saved final CSV.")