import glob
import pandas as pd

eval_id = "120k"

files = sorted(glob.glob(f"{eval_id}_eval/masseval_{eval_id}_part_*.csv"))
print("Found files:", len(files))

for f in files:
    df = pd.read_csv(f, low_memory=False)
    print("\nFile:", f)
    print("Rows:", len(df))
    
    for col in ["plddt", "progres", "scaccuracy", "seq_id", "seq_similarity"]:
        if col in df.columns:
            print(f"  {col} non-null:", df[col].notna().sum())

prefiltered = pd.read_csv(f"{eval_id}_eval/{eval_id}_eval_seqs_prefiltered.csv")
print(prefiltered.info())


