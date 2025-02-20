import os
import subprocess
batch_size = 4

def calculate_max_tm_score(query_pdb, pdb_database_path):
    max_tm_score = 0.0
    
    # Iterate over each PDB entry in the database
    for pdb_entry in os.listdir(pdb_database_path):
        pdb_entry_path = os.path.join(pdb_database_path, pdb_entry)
        
        # Run TM-align and capture the output
        result = subprocess.run(
            ['./TMalign', query_pdb, pdb_entry_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Parse TM-score from TM-align output
        for line in result.stdout.split("\n"):
            if "TM-score=" in line:
                tm_score = float(line.split()[1])  # Extract TM-score
                max_tm_score = max(max_tm_score, tm_score)  # Update max TM-score

    return max_tm_score

# Directory containing PDB files
pdb_database_path = "./pdb_database/"

# Loop over generated sequences and calculate max TM-scores
for i in range(batch_size):
    query_pdb = f"./validation/generated_protein_structure_{i}.pdb"
    max_tm_score = calculate_max_tm_score(query_pdb, pdb_database_path)
    print(f"Max TM-score for sequence {i}: {max_tm_score:.3f}")


