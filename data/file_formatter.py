import random

input_file = '../../data/filtered_2_1_1_37.txt'
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

with open('dnmt_critic.txt', "w") as critic_file:
    for seq in critic_sequences:
        seq = " ".join(seq)
        critic_file.write(seq + "\n")

with open('dnmt_full.txt', "w") as whole_file:
    for seq in sequences:
        whole_file.write(seq + "\n")

print(f"Generator set size: {len(gen_sequences)}")
print(f"Critic set size: {len(critic_sequences)}")

mini_gen_sequences = gen_sequences[:100]
mini_critic_sequences = critic_sequences[:100]

with open('dnmt_gen_mini.txt', "w") as mini_gen_file:
    for seq in mini_gen_sequences:
        if len(seq) < 128:
            seq = " ".join(seq)
            mini_gen_file.write(seq + "\n")

with open('dnmt_critic_mini.txt', "w") as mini_critic_file:
    for seq in mini_critic_sequences:
        if len(seq) < 128:
            seq = " ".join(seq)
            mini_critic_file.write(seq + "\n")