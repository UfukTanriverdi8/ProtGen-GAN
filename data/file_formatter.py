import pandas as pd
import random

csv_file = 'IPR001525_dnmt3a_dataset_filtered.xlsx - Sheet1.csv'

# Read CSV and convert the last column to a Python list
df = pd.read_csv(csv_file, header=None)
sequences = df.iloc[:, -1].tolist()

# Shuffle the list in-place
random.shuffle(sequences)

# Split 90% train, 10% validation
val_size = int(len(sequences) * 0.1)
val_data = sequences[:val_size]
train_data = sequences[val_size:]

# Write train/val files
with open("dnmt_train.txt", "w") as f_train:
    for seq in train_data:
        f_train.write(" ".join(str(seq)) + "\n")

with open("dnmt_val.txt", "w") as f_val:
    for seq in val_data:
        f_val.write(" ".join(str(seq)) + "\n")

# Write unformatted/full files
with open('dnmt_unformatted.txt', 'w') as f:
    for seq in sequences:
        f.write(str(seq).strip() + "\n")

with open("dnmt_full.txt", "w") as f:
    for seq in sequences:
        f.write(" ".join(str(seq)) + "\n")


split_index = int(len(sequences) * 0.5)
gen_sequences = sequences[:split_index]
critic_sequences = sequences[split_index:]
with open('dnmt_gen.txt', 'w') as train_file:
    for seq in gen_sequences:
        seq = " ".join(seq)
        train_file.write(seq + '\n')

with open('dnmt_critic.txt', "w") as critic_file:
    for seq in critic_sequences:
        seq = " ".join(seq)
        critic_file.write(seq + "\n")
