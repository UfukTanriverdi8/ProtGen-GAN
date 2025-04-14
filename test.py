import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, EsmForProteinFolding
from models import Generator, Critic
import os
import wandb
from loss import critic_loss, generator_loss, compute_gradient_penalty
from dataset import load_and_tokenize_dataset, get_dataloaders
from torch.optim import AdamW
#from val_metrics import calculate_plddt_scores_and_save_pdb, calculate_tm_scores, clean_m8_folder, calculate_mpnn_alignment_metric, generate_fake_sequences

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device="cuda"

from tmtools.io import get_structure, get_residue_data
from tmtools import tm_align

# Load the first protein structure and extract backbone data
structure1 = get_structure("validation/test_pdb/generated_protein_0.pdb")
chains = structure1.get_chains()
chain1 = next(chains)
print(chain1)
coords1, seq1 = get_residue_data(chain1)
print(seq1)

# Do the same for the second protein
structure2 = get_structure("validation/pdbs/generated_protein_1.pdb")
chain2 = next(structure2.get_chains())
coords2, seq2 = get_residue_data(chain2)

# Compare the proteins using their backbone coordinates and sequences
result = tm_align(coords1, coords2, seq1, seq2)

# Print the TM score and alignment details
print("TM score (chain1):", result.tm_norm_chain1)
print("TM score (chain2):", result.tm_norm_chain2)
print("RMSD:", result.rmsd)




""" model_checkpoint_path = f"../checkpoints/dynamic/saved-final-300"
generator_protbert = AutoModelForMaskedLM.from_pretrained(model_checkpoint_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path, do_lower_case=False)


generator = Generator(generator_protbert, tokenizer).to(device)
esmfold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).to("cuda")
esmfold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
esmfold_model.esm = esmfold_model.esm.half()
esmfold_model.eval()


generated_sequences = generate_fake_sequences(
    generator=generator,
    tokenizer=tokenizer,
    num_sequences=10,
)

calculate_plddt_scores_and_save_pdb(
    generated_sequences=generated_sequences,
    folding_model=esmfold_model,
    folding_tokenizer=esmfold_tokenizer,
    num_sequences=10,
)

calculate_mpnn_alignment_metric(
    generated_sequences=generated_sequences,
) """