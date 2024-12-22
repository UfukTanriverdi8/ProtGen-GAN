import numpy as np
from collections import Counter

# Load the sequences and calculate their lengths
file_name = "dnmt_full.txt"
with open(file_name, "r") as f:
    lines = f.readlines()

lines = [line.strip() for line in lines]
sequence_lengths = [len(line) for line in lines]

more_lengths = []

for i in sequence_lengths:
    if i > 512:
        more_lengths.append(i)

for i in more_lengths:
    print(i)