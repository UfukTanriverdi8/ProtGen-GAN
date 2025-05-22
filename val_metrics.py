import random
import os
import torch
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
import requests
import time
import subprocess
import concurrent.futures
import tarfile
import shutil
import numpy as np

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def clean_m8_folder():
    """ Deletes all files and subdirectories inside validation/m8s/. """
    m8_folder = "validation/m8s/"
    
    if os.path.exists(m8_folder):
        shutil.rmtree(m8_folder)  
        os.makedirs(m8_folder, exist_ok=True)  
        print("🧹 Cleaned up validation/m8s/ folder.")
    else:
        print("⚠️ validation/m8s/ folder does not exist.")


def sample_sequence_length(file_path="data/dnmt_unformatted.txt", variation=0.1):
    with open(file_path, 'r') as file:
        sequence_lengths = [len(line.strip()) for line in file if line.strip()]
    base_length = random.choice(sequence_lengths)
    variation_amount = int(base_length * variation)
    sampled_length = base_length + random.randint(-variation_amount, variation_amount)
    sampled_length = max(1, sampled_length)
    if sampled_length > 500:
        sampled_length = 500
    return sampled_length

def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

def generate_fake_sequences(generator, tokenizer, num_sequences, device="cuda"):
    """
    Generates fake protein sequences using the generator model.
    
    Args:
        generator: The generator model (e.g., a fine-tuned ProtBERT).
        tokenizer: The tokenizer for the model.
        num_sequences: How many sequences to generate.
        device: The device to run the generation on.
    
    Returns:
        A list of generated sequences (as strings).
    """
    generated_sequences = []
    for _ in range(num_sequences):
        # Sample the sequence length from the dataset
        sample_length = sample_sequence_length('./data/dnmt_unformatted.txt')
        
        # Create a fully masked input for sequence generation
        fully_masked_input = torch.full((1, sample_length + 2), generator.mask_token_id).to(device)
        fully_masked_input[0, 0] = tokenizer.cls_token_id  # [CLS]
        fully_masked_input[0, -1] = tokenizer.sep_token_id  # [SEP]
        attention_mask = torch.ones_like(fully_masked_input).to(device)
        
        # Iteratively fill the sequence
        current_masking_rate = 1.0
        iteration_fill_rate = 0.1
        while current_masking_rate > 0:
            random_temp = torch.empty(1).uniform_(0.8, 1.2).item()  # Random temperature
            with torch.no_grad():
                generated_ids = generator.generate(
                    fully_masked_input,
                    attention_mask,
                    keep_percent=iteration_fill_rate,
                    current_rate=current_masking_rate,
                    temperature=random_temp
                )
            fully_masked_input = generated_ids.clone()
            current_masking_rate = max(0, current_masking_rate - iteration_fill_rate)
        
        # Decode the generated sequence and clean it
        decoded_sequence = tokenizer.decode(fully_masked_input[0], skip_special_tokens=True).replace(" ", "")[:600]
        generated_sequences.append(decoded_sequence)
    
    return generated_sequences


def calculate_plddt_scores_and_save_pdb(generated_sequences, folding_tokenizer, folding_model, num_sequences=10, batch_size=4, device="cuda"):
    """
    Generates fake sequences, folds them using ESMFold, converts outputs to PDB files,
    and computes the average pLDDT score.
    """
    # Step 1: Generate sequences using the separated function.    
    print("*" * 10 + " Generated Sequences " + "*" * 10)
    print(generated_sequences)
    print("📊 Sequence lengths:", [len(seq) for seq in generated_sequences])
    
    # Step 2: Process sequences in batches for ESMFold
    plddt_scores = []
    os.makedirs("./validation/pdbs", exist_ok=True)
    
    for batch_start in range(0, len(generated_sequences), batch_size):
        batch_sequences = generated_sequences[batch_start:batch_start + batch_size]
    
        # Tokenize the batch for ESMFold
        tokenized_inputs = folding_tokenizer(
            batch_sequences,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True
        )['input_ids'].to(device)
    
        with torch.no_grad():
            outputs = folding_model(tokenized_inputs)
    
        # Convert outputs to PDB using the existing function
        pdb_list = convert_outputs_to_pdb(outputs)
    
        # Save each PDB file and calculate the pLDDT score
        for i, pdb_data in enumerate(pdb_list):
            pdb_filename = f"./validation/pdbs/generated_protein_{batch_start + i}.pdb"
            with open(pdb_filename, "w") as f:
                f.write(pdb_data)
            print(f"Saved PDB: {pdb_filename}")
    
            # Compute average pLDDT score for this sequence
            average_plddt = outputs["plddt"][i].mean().item()
            plddt_scores.append(average_plddt)
    
    print(f"All pLDDT scores: {plddt_scores}")
    del tokenized_inputs, outputs, pdb_list
    torch.cuda.empty_cache()
    
    # Return the average pLDDT score across all sequences
    return sum(plddt_scores) / len(plddt_scores)


# ---------- Foldseek helpers (robust version) ----------
import os, time, tarfile, subprocess, requests
from urllib3.exceptions import NameResolutionError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

FOLDSEEK_API = "https://search.foldseek.com/api"
PDB_FOLDER   = "validation/pdbs/"
M8_FOLDER    = "validation/m8s/"

# 1) one session with automatic retries on DNS + 5xx + timeouts
retry_cfg = Retry(
    total=5,                # retry up to 5×
    backoff_factor=1.0,     # 1 s, 2 s, 4 s, 8 s, 16 s …
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["GET", "POST"]
)
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retry_cfg))

# 2) submit a PDB and get a ticket
def submit_to_foldseek(pdb_file, wait_time=30):
    while True:
        try:
            with open(pdb_file, "rb") as fh:
                r = session.post(
                    f"{FOLDSEEK_API}/ticket",
                    files={"q": fh},
                    data={
                        "mode": "tmalign",
                        "database[]": ["afdb50", "afdb-swissprot",
                                       "afdb-proteome", "cath50"]
                    },
                    timeout=30
                )
            r.raise_for_status()
            tid = r.json()["id"]
            print(f"✅  submitted {pdb_file}  → ticket {tid}")
            return tid
        except (requests.ConnectionError, NameResolutionError,
                requests.Timeout) as e:
            print(f"⚠️  {pdb_file}: {e} – retry in {wait_time}s")
            time.sleep(wait_time)

# 3) poll tickets until they are COMPLETE
def wait_for_completion(tickets, poll=10, max_wait=600):
    start = time.time()
    pending = set(tickets)
    while pending and time.time() - start < max_wait:
        for tid in list(pending):
            try:
                r = session.get(f"{FOLDSEEK_API}/ticket/{tid}", timeout=15)
                r.raise_for_status()
                if r.json().get("status") == "COMPLETE":
                    print(f"🏁  {tid} done")
                    pending.remove(tid)
            except (requests.ConnectionError, NameResolutionError,
                    requests.Timeout) as e:
                print(f"⚠️  poll {tid}: {e}")
        if pending:
            time.sleep(poll)
    if pending:
        print("❌  timeout for", pending)
    return list(set(tickets) - pending)          # completed list

# 4) download + untar results, returns list of *.m8 files
def download_results(ticket, out_dir, max_tries=5, backoff=20):
    url      = f"{FOLDSEEK_API}/result/download/{ticket}"
    tar_path = os.path.join(out_dir, f"{ticket}.tar")
    job_dir  = os.path.join(out_dir, ticket)
    os.makedirs(job_dir, exist_ok=True)

    for n in range(1, max_tries + 1):
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            with open(tar_path, "wb") as fh:
                fh.write(r.content)

            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(job_dir)

            m8s = [os.path.join(job_dir, f)
                   for f in os.listdir(job_dir) if f.endswith(".m8")]
            if not m8s:
                print(f"❌  {ticket}: no .m8 files")
                return None
            print(f"📦  {ticket}: downloaded & extracted")
            return m8s
        except (requests.ConnectionError, NameResolutionError,
                requests.Timeout) as e:
            print(f"⚠️  {ticket}: {e} (retry {n}/{max_tries})")
            time.sleep(backoff * n)
        except Exception as e:   # tarfile or other I/O errors
            print(f"❌  {ticket}: {e}")
            return None
    print(f"❌  {ticket}: gave up after {max_tries} tries")
    return None

# 5) extract highest TM‑score in a *.m8 bundle
def extract_max_tm_score(m8_files, ticket):
    max_tm = 0.0
    for path in m8_files:
        try:
            res = subprocess.run(
                f"awk -F'\t' '{{print $12}}' {path} | sort -nr | head -1",
                shell=True, capture_output=True, text=True
            )
            s = res.stdout.strip()
            if s:
                val = float(s)
                max_tm = max(max_tm, val)
            print(f"🔍  {ticket}: {os.path.basename(path)} → {s}")
        except Exception as e:
            print(f"❌  {ticket}: {e}")
    return max_tm

# 6) main driver – unchanged signature
def calculate_tm_scores(num_sequences=10):
    pdbs   = [os.path.join(PDB_FOLDER, f"generated_protein_{i}.pdb")
              for i in range(num_sequences)]

    # submit
    tickets = {submit_to_foldseek(p): p for p in pdbs}

    # wait
    done = wait_for_completion(tickets.keys())
    if not done:
        return None

    # download results
    all_m8 = {t: download_results(t, M8_FOLDER) for t in done}
    all_m8 = {k: v for k, v in all_m8.items() if v}

    # extract TM
    scores = [extract_max_tm_score(v, k) for k, v in all_m8.items()]
    return sum(scores) / len(scores) if scores else None
# -------------------------------------------------------



#calculate_tm_scores()

from ProteinMPNN.protein_mpnn_utils import (
    parse_PDB,
    _S_to_seq,
    tied_featurize,
    StructureDatasetPDB,
    ProteinMPNN,
)

def get_mpnn_sequence_from_pdb(
    pdb_file,
    device="cuda",
    ca_only=False,
    model_weights_path=None,
    model_name="v_48_020",
    max_length=512,
    temperature=0.1
):
    """
    Given a PDB file, this function uses ProteinMPNN to design a sequence from the structure.
    Returns the designed sequence as a string.
    """
    # Parse the PDB file into a dictionary list
    pdb_dict_list = parse_PDB(pdb_file, ca_only=ca_only)
    if not pdb_dict_list:
        print("No valid chains found in the provided PDB.")
        return ""
    
    # Create a dataset using the parsed PDB information
    dataset = StructureDatasetPDB(pdb_dict_list, max_length=max_length)
    # Use the first entry as our single example batch
    batch = [dataset[0]]
    
    # Construct a chain_id_dict from available chains in the PDB
    chains = []
    for key in pdb_dict_list[0]:
        if key.startswith("seq_chain_"):
            chains.append(key[len("seq_chain_"):])
    if chains:
        chain_id_dict = {pdb_dict_list[0]['name']: (chains, [])}
    else:
        chain_id_dict = {}
    
    # Optional dictionaries (not used here)
    fixed_positions_dict = None
    omit_AA_dict = None
    tied_positions_dict = None
    pssm_dict = None
    bias_by_res_dict = None

    # Prepare input tensors for ProteinMPNN using tied_featurize
    X, S, mask, lengths, chain_M, chain_encoding_all, _, _, _, _, \
        chain_M_pos, omit_AA_mask, residue_idx, _, _, \
        pssm_coef_all, pssm_bias_all, pssm_log_odds_all, \
        bias_by_res_all, tied_beta = tied_featurize(
            batch, device, chain_id_dict, fixed_positions_dict,
            omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict,
            ca_only=ca_only
        )
    
    # Load the ProteinMPNN model
    if model_weights_path is None:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_weights_path = os.path.join(current_dir,"ProteinMPNN", "vanilla_model_weights")
    if not model_weights_path.endswith(os.path.sep):
        model_weights_path += os.path.sep
    checkpoint_path = os.path.join(model_weights_path, f"{model_name}.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Set model hyperparameters (should match those used during training)
    num_letters = 21
    node_features = 128
    edge_features = 128
    hidden_dim = 128
    num_encoder_layers = 3
    num_decoder_layers = 3
    k_neighbors = checkpoint['num_edges']
    augment_eps = checkpoint.get('noise_level', 0.05)
    
    model = ProteinMPNN(
        num_letters, node_features, edge_features, hidden_dim,
        num_encoder_layers, num_decoder_layers,
        k_neighbors=k_neighbors, augment_eps=augment_eps, ca_only=ca_only
    )
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create a random noise tensor for sampling
    randn = torch.randn(X.shape[0], X.shape[1], device=device)
    # For simplicity, we use zeros for omit_AAs and bias
    omit_AAs_np = np.zeros(21, dtype=np.float32)
    bias_AAs_np = np.zeros(21, dtype=np.float32)
    pssm_log_odds_mask = np.ones((X.shape[1], 21), dtype=np.float32)
    
    # Sample a sequence using ProteinMPNN
    sample_output = model.sample(
        X, randn, S, chain_M, chain_encoding_all, residue_idx, mask=mask,
        temperature=temperature, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np,
        chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef_all,
        pssm_bias=pssm_bias_all, pssm_multi=0.0, pssm_log_odds_flag=0,
        pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=0, bias_by_res=bias_by_res_all
    )
    
    S_sample = sample_output["S"]  # [batch, seq_length]
    # Convert the sampled sequence tensor into a string
    designed_seq = _S_to_seq(S_sample[0].cpu().numpy(), mask[0].cpu().numpy())
    
    return designed_seq


""" pdb_path = "validation/test_pdb/generated_protein_0.pdb"
seq = get_mpnn_sequence_from_pdb(pdb_path)
print("Predicted Sequence:", seq) """

seq1 = "MADKQLEFICPVSTGNRYWD"
seq2 = "MADKKLEFICPVSTGNRYWA"

from Bio.Align import PairwiseAligner
from Bio import pairwise2
from Bio.Align.substitution_matrices import load

blosum62 = load("BLOSUM62")

def compute_alignment_identity_new(seq1, seq2):
    # Initialize the aligner
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.substitution_matrix = blosum62

    # Perform the alignment; this returns an Alignment object
    alignments = aligner.align(seq1, seq2)
    best_alignment = alignments[0]

    # Each alignment object has .aligned which is a tuple of two lists:
    # (intervals for seq1, intervals for seq2). Each interval is (start, end).
    intervals1, intervals2 = best_alignment.aligned

    def reconstruct_alignment(seq, intervals):
        aligned_seq = []
        prev_end = 0
        for start, end in intervals:
            # Add gaps for unaligned region
            if start > prev_end:
                aligned_seq.append("-" * (start - prev_end))
            aligned_seq.append(seq[start:end])
            prev_end = end
        # Add trailing gaps if any
        if prev_end < len(seq):
            aligned_seq.append("-" * (len(seq) - prev_end))
        return "".join(aligned_seq)
    
    aligned_seq1 = reconstruct_alignment(seq1, intervals1)
    aligned_seq2 = reconstruct_alignment(seq2, intervals2)
    
    # If the lengths don't match (should rarely happen), pad the shorter one with gaps.
    if len(aligned_seq1) != len(aligned_seq2):
        max_len = max(len(aligned_seq1), len(aligned_seq2))
        aligned_seq1 = aligned_seq1.ljust(max_len, '-')
        aligned_seq2 = aligned_seq2.ljust(max_len, '-')
    
    # Compute percent identity
    matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b)
    identity = matches / len(aligned_seq1)
    
    return identity, aligned_seq1, aligned_seq2

def compute_alignment_identity(seq1, seq2):
    # Set up the alignment parameters using BLOSUM62
    # gap_open is the penalty for opening a gap
    # gap_extend is the penalty for extending a gap
    gap_open = -10.0
    gap_extend = -0.5
    
    # Perform global alignment with BLOSUM62
    # This returns a list of alignments sorted by score
    alignments = pairwise2.align.globalds(
        seq1, 
        seq2, 
        blosum62,          # Substitution matrix
        gap_open,          # Gap open penalty
        gap_extend,        # Gap extension penalty
        one_alignment_only=True  # Only return the best alignment
    )
    
    # Get the best alignment
    best_alignment = alignments[0]
    aligned_seq1, aligned_seq2, score, start, end = best_alignment
    
    # Compute percent identity by comparing aligned sequences
    matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b and a != '-' and b != '-')
    aligned_positions = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a != '-' or b != '-')
    
    # Calculate identity (matched positions / aligned positions)
    identity = matches / aligned_positions if aligned_positions > 0 else 0.0
    
    return identity, aligned_seq1, aligned_seq2



def calculate_mpnn_alignment_metric(generated_sequences, num_sequences=10, batch_size=4, device="cuda"):
    """ generated_sequences = generate_fake_sequences(generator, tokenizer, num_sequences, device)
    print("Generated sequences:")
    for idx, seq in enumerate(generated_sequences):
        print(f"Sequence {idx}: {seq}") """
    
    
    alignment_identities = []
    for i, gen_seq in enumerate(generated_sequences):
        pdb_file = f"./validation/pdbs/generated_protein_{i}.pdb"
        if not os.path.exists(pdb_file):
            print(f"PDB file {pdb_file} not found. Skipping.")
            continue
        predicted_seq = get_mpnn_sequence_from_pdb(pdb_file, device=device)
        identity, aligned_gen, aligned_pred = compute_alignment_identity(gen_seq, predicted_seq)
        print(f"\nSequence {i}:")
        print("Generated Sequence:\n", gen_seq)
        print("Predicted Sequence:\n", predicted_seq)
        #print("Aligned Generated:\n", aligned_gen)
        #print("Aligned Predicted:\n", aligned_pred)
        print("Alignment Identity: {:.2%}".format(identity))
        alignment_identities.append(identity)
    
    if alignment_identities:
        avg_identity = sum(alignment_identities) / len(alignment_identities)
        print("\nAverage ProteinMPNN Alignment Identity: {:.2%}".format(avg_identity))
    else:
        avg_identity = 0
        print("No alignments were computed.")
    
    return avg_identity


import progres as pg
def compute_average_progres_score(reference_pdb = "validation/pdbs/reference/DNMT3A.pdb", num_files = 10):
    scores = []
    for i in range(num_files):
        gen_file = f"validation/pdbs/generated_protein_{i}.pdb"
        score = pg.progres_score(reference_pdb, gen_file)
        scores.append(score)

    #print(scores)
    return sum(scores) / len(scores) if scores else 0.0

from tmtools.io import get_structure, get_residue_data
from tmtools import tm_align

def calculate_pairwise_tm_score():
    """
    Calculate TM-score between all pairs of generated PDBs.
    """
    pdb_files = [os.path.join(PDB_FOLDER, f"generated_protein_{i}.pdb") for i in range(10)]
    tm_scores = []
    
    for i in range(len(pdb_files)):
        for j in range(i + 1, len(pdb_files)):
            pdb1 = pdb_files[i]
            pdb2 = pdb_files[j]
            structure1 = get_structure(pdb1)
            structure2 = get_structure(pdb2)
            coords1, seq1 = get_residue_data(next(structure1.get_chains()))
            coords2, seq2 = get_residue_data(next(structure2.get_chains()))
            result = tm_align(coords1, coords2, seq1, seq2)
            tm_scores.append((result.tm_norm_chain1 + result.tm_norm_chain2) / 2)
    
    avg_pairwise_tm_score = sum(tm_scores) / len(tm_scores) if tm_scores else 0.0
    return avg_pairwise_tm_score

