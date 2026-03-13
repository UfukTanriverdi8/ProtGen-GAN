import os
import glob
from collections import Counter

COLLAPSE_SEQ = "MRTIDLFAGCGGLSLGFQNAGFDIVAAFENWIPAIKVYQRNFRHPIFVDLSRESELAEYNPDIIVGGPPCQDFSSAGKRDESLGRANLTITFAIIASVKPQWFVMENVDRITKSILPKAKQIFKNGYGLTGSVLNSSYCGVPQARKRYFLIGELEGEDDVLEYQLLETQSQSKPMTVFDYLGNELGIEYFYRHPRSYMRRAIFSIYEPSPTIRGVNRPIPKTYKKHPGDACDLNESLRPLTTRERAYIQTFPKFKFEGNKSDLEQMIGNAVPVKLAEYIAKCILQYLEDK"

txt_files = sorted(glob.glob("raw_out_120k_generation/*.txt"))
if not txt_files:
    print("No .txt files found in current directory.")
    exit()

print(f"Found {len(txt_files)} txt files\n")

for path in txt_files:
    seqs = [l.strip().upper() for l in open(path) if l.strip()]
    total = len(seqs)
    unique = len(set(seqs))
    suspect_count = seqs.count(COLLAPSE_SEQ)
    top = Counter(seqs).most_common(3)

    print(f"--- {path} ---")
    print(f"  Total: {total} | Unique: {unique} | Duplicates: {total - unique}")
    print(f"  Suspect seq: {suspect_count} times")
    print(f"  Top 3 most common: {[(count, seq[:40]+'...') for seq, count in top]}")
    print()