import random

input_file = 'filtered_2_1_1_37.txt'
sequences = []

count = 0

with open(input_file, 'r') as file:
    next(file)
    for line in file:
        columns = line.strip().split('\t')

        sequence = columns[-1]

        if sequence.startswith('https'):
            count += 1
            continue
        sequence = sequence.replace('U', 'X').replace('B', 'X').replace('Z', 'X').replace('O', 'X')

        sequences.append(sequence)

print(f"{count} rows skipped!")
random.shuffle(sequences)

train_ratio = 0.5
split_index = int(len(sequences) * train_ratio)

gen_sequences = sequences[:split_index]
critic_sequences = sequences[split_index:]

with open('dnmt_gen.txt', 'w') as train_file:
    for seq in gen_sequences:
        seq = " ".join(seq)
        train_file.write(seq + '\n')

with open('dnmt_critic.txt', "w") as val_file:
    for seq in critic_sequences:
        seq = " ".join(seq)
        val_file.write(seq + "\n")

with open('dnmt_full.txt', "w") as whole_file:
    for seq in sequences:
        whole_file.write(seq + "\n")

print(f"Generator set size: {len(gen_sequences)}")
print(f"Critic set size: {len(critic_sequences)}")