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
from ProteinMPNN.protein_mpnn_utils import (
    parse_PDB,
    _S_to_seq,
    tied_featurize,
    StructureDatasetPDB,
    ProteinMPNN,
)
from tmtools.io import get_structure, get_residue_data
from tmtools import tm_align
from Bio.Align import PairwiseAligner
from Bio.Align.substitution_matrices import load
import progres as pg


_MPNN_CACHE = {}


blosum62 = load("BLOSUM62")

test_seq1 = "MADKQLEFICPVSTGNRYWD"
test_seq2 = "MADKKLEFICPVSTGNRYWA"


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
        try:
            aa = outputs["aatype"][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = outputs["residue_index"][i].astype(np.int32) + 1
            if "chain_index" in outputs and outputs["chain_index"] is not None:
                chain_idx_i = outputs["chain_index"][i]
                # replace NaNs/None with 0, then int
                if chain_idx_i.dtype.kind in ("f", "O"):
                    # NaN check only valid for float
                    if chain_idx_i.dtype.kind == "f" and np.isnan(chain_idx_i).any():
                        chain_idx_i = np.zeros_like(resid, dtype=np.int32)
                chain_idx_i = chain_idx_i.astype(np.int32, copy=False)
            else:
                chain_idx_i = np.zeros_like(resid, dtype=np.int32)
            pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=chain_idx_i,
            )
            pdbs.append(to_pdb(pred))
        except Exception as e:
            # Log and append a sentinel so caller can mark this item as failed without killing the batch
            print(f"[WARN] to_pdb failed for batch item {i}: {e}")
            pdbs.append(None)
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
        fixed_temp = 1.0 
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
                    temperature=fixed_temp
                )
            fully_masked_input = generated_ids.clone()
            current_masking_rate = max(0, current_masking_rate - iteration_fill_rate)
        
        # Decode the generated sequence and clean it
        decoded_sequence = tokenizer.decode(fully_masked_input[0], skip_special_tokens=True).replace(" ", "")[:600]
        generated_sequences.append(decoded_sequence)
    
    return generated_sequences


def calculate_plddt_scores_and_save_pdb(generated_sequences, folding_tokenizer, folding_model, num_sequences=10, batch_size=4, device="cuda", run_name="default_run_name"):
    """
    Generates fake sequences, folds them using ESMFold, converts outputs to PDB files,
    and computes the average pLDDT score.
    """
    # Step 1: Generate sequences using the separated function.    
    """ print("*" * 10 + " Generated Sequences " + "*" * 10)
    print(generated_sequences) """
    print("📊 Sequence lengths:", [len(seq) for seq in generated_sequences])
    
    # Step 2: Process sequences in batches for ESMFold
    plddt_scores = []
    pdb_folder = f"./validation/pdbs/{run_name}"
    os.makedirs(pdb_folder, exist_ok=True)
    
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
            pdb_filename = f"{pdb_folder}/generated_protein_{batch_start + i}.pdb"
            if pdb_data is None:
                print(f"[WARN] skipping PDB write for item {batch_start + i} in the batch number of {batch_start/batch_size} (pdb_data=None)")
                # DO NOT write the file. Just continue; the caller will handle missing files.
                continue
            # no name change anymore
            """ if os.path.exists(pdb_filename):
                pdb_filename = pdb_filename.split(".pdb")[0] + str(custom_row_id) + ".pdb" """
            with open(pdb_filename, "w") as f:
                f.write(pdb_data)
            print(f"Saved PDB: {pdb_filename}")
    
            # Compute average pLDDT score for this sequence
            average_plddt = outputs["plddt"][i].mean().item()
            plddt_scores.append(average_plddt)
    
    print(f"All pLDDT scores: {plddt_scores}")
    try:
        del tokenized_inputs, outputs, pdb_list
    except Exception:
        pass
    torch.cuda.empty_cache()
    
    valid = [x for x in plddt_scores if isinstance(x, (float, int)) and not np.isnan(x)]
    avg_plddt_score = (sum(valid) / len(valid)) if valid else -1
    # Return the average pLDDT score across all sequences
    if len(plddt_scores) > 0:
        avg_plddt_score = sum(plddt_scores) / len(plddt_scores)
    else:
        avg_plddt_score = -1
    return avg_plddt_score, plddt_scores



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
    cache_key = (str(device), ca_only, checkpoint_path)
    if cache_key in _MPNN_CACHE:
        model, checkpoint = _MPNN_CACHE[cache_key]
    else:
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
    
    _MPNN_CACHE[cache_key] = (model, checkpoint)
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



def compute_alignment_identity_and_similarity(seq1, seq2):
    """
    Compute both sequence identity and similarity using Biopython's PairwiseAligner.
    - Identity: exact matches / alignment length
    - Similarity: (matches + conservative substitutions) / alignment length
    """
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.substitution_matrix = blosum62

    alignments = aligner.align(seq1, seq2)
    best_alignment = alignments[0]

    aligned_seq1, aligned_seq2 = str(best_alignment[0]), str(best_alignment[1])

    matches, similar = 0, 0
    for a, b in zip(aligned_seq1, aligned_seq2):
        if a == b:
            matches += 1
            similar += 1
        elif a != "-" and b != "-":
            # if substitution score > 0, consider it similar
            score = blosum62[a, b]
            if score > 0:
                similar += 1

    alignment_length = len(aligned_seq1)
    identity = matches / alignment_length
    similarity = similar / alignment_length

    return identity, similarity, aligned_seq1, aligned_seq2


def calculate_mpnn_alignment_metric(generated_sequences, num_sequences=10, batch_size=4, device="cuda", run_name="default_run_name"):
    """ generated_sequences = generate_fake_sequences(generator, tokenizer, num_sequences, device)
    print("Generated sequences:")
    for idx, seq in enumerate(generated_sequences):
        print(f"Sequence {idx}: {seq}") """
    pdb_folder = f"./validation/pdbs/{run_name}"
    os.makedirs(pdb_folder, exist_ok=True)
    
    
    alignment_identities = []
    for i, gen_seq in enumerate(generated_sequences):
        pdb_file = f"{pdb_folder}/generated_protein_{i}.pdb"
        if not os.path.exists(pdb_file):
            print(f"PDB file {pdb_file} not found. Skipping.")
            continue
        predicted_seq = get_mpnn_sequence_from_pdb(pdb_file, device=device)
        identity, similarity,  aligned_gen, aligned_pred = compute_alignment_identity_and_similarity(gen_seq, predicted_seq)
        print(f"\nSequence {i}:")
        print("Generated Sequence:\n", gen_seq)
        print("Predicted Sequence:\n", predicted_seq)
        print("Alignment Identity: {:.2%}".format(identity))
        alignment_identities.append(identity)
    
    if alignment_identities:
        avg_identity = sum(alignment_identities) / len(alignment_identities)
        print("\nAverage ProteinMPNN Alignment Identity: {:.2%}".format(avg_identity))
    else:
        avg_identity = 0
        print("No alignments were computed.")
    
    return avg_identity


def compute_average_progres_score(reference_pdb = "validation/pdbs/reference/DNMT3A.pdb", num_sequences = 10, run_name="default_run_name"):
    scores = []
    pdb_folder = f"./validation/pdbs/{run_name}"
    os.makedirs(pdb_folder, exist_ok=True)

    for i in range(num_sequences):
        gen_file = f"{pdb_folder}/generated_protein_{i}.pdb"
        score = pg.progres_score(reference_pdb, gen_file)
        scores.append(score)

    #print(scores)
    return sum(scores) / len(scores) if scores else 0.0



def calculate_pairwise_tm_score(run_name="default_run_name", num_sequences=10):
    """
    Calculate TM-score between all pairs of generated PDBs.
    """
    pdb_folder = f"./validation/pdbs/{run_name}"
    os.makedirs(pdb_folder, exist_ok=True)
    pdb_files = [os.path.join(pdb_folder, f"generated_protein_{i}.pdb") for i in range(num_sequences)]
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


# ------ Foldseek API Configuration for max tm score metric ------
FOLDSEEK_API = "https://search.foldseek.com/api"
PDB_FOLDER = "validation/pdbs/"
M8_FOLDER = "validation/m8s/"

def submit_to_foldseek(pdb_file, wait_time=120):
    while True:
        try:
            with open(pdb_file, 'rb') as file:
                response = requests.post(
                    f"{FOLDSEEK_API}/ticket",
                    files={"q": file},
                    data={'mode': 'tmalign', 'database[]': ['afdb50', 'afdb-swissprot', 'afdb-proteome', 'cath50']}
                )
            if response.status_code == 200:
                ticket = response.json().get("id")
                print(f"✅ Submitted {pdb_file} | Ticket ID: {ticket}")
                return ticket
            else:
                print(f"❌ Submission failed for {pdb_file}: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"🪝 Catched the error: {e}")
            print(f"⚠️ Error submitting: {pdb_file}")
        print(f"⏳ Retrying in {wait_time} seconds...")
        time.sleep(wait_time)


def wait_for_completion(tickets, interval=10, max_wait=600):
    """ Waits for all Foldseek jobs to complete. """
    start_time = time.time()
    pending_tickets = set(tickets)

    while time.time() - start_time < max_wait and pending_tickets:
        print(f"⏳ Checking status for {len(pending_tickets)} jobs...")

        completed_tickets = []
        for ticket in list(pending_tickets):
            try:
                response = requests.get(f"{FOLDSEEK_API}/ticket/{ticket}", timeout=10)
                response.raise_for_status()  # Raise error for non-200 responses
                status = response.json().get("status")

                if status == "COMPLETE":
                    print(f"✅ Job {ticket} completed!")
                    completed_tickets.append(ticket)
                else:
                    print(f"🔄 Job {ticket} still running...")
            
            except Exception as e:
                print(f"⚠️ API error for {ticket}: {e} (Retrying in {interval*20}s)")
                time.sleep(interval*20)  # Wait before retrying
                continue

        for ticket in completed_tickets:
            pending_tickets.remove(ticket)

        if pending_tickets:
            time.sleep(interval)

    if pending_tickets:
        print(f"❌ Timeout waiting for jobs: {pending_tickets}")

    return list(set(tickets) - pending_tickets)  # Return completed tickets


def download_results(ticket, output_folder):
    """ Downloads and extracts Foldseek results (.m8) for the given ticket. """
    tar_file = os.path.join(output_folder, f"{ticket}.tar")
    ticket_folder = os.path.join(output_folder, ticket)
    os.makedirs(ticket_folder, exist_ok=True)

    response = requests.get(f"{FOLDSEEK_API}/result/download/{ticket}")

    if response.status_code == 200:
        with open(tar_file, "wb") as f:
            f.write(response.content)
        print(f"✅ Downloaded results for {ticket}")
    else:
        print(f"❌ Failed to download results for {ticket}: {response.text}")
        return None

    # Extract .tar archive
    try:
        with tarfile.open(tar_file, "r") as tar:
            tar.extractall(ticket_folder)
        print(f"📂 Extracted files for {ticket}")
    except Exception as e:
        print(f"❌ Error extracting .tar for {ticket}: {e}")
        return None

    # Find all .m8 files
    m8_files = [os.path.join(ticket_folder, f) for f in os.listdir(ticket_folder) if f.endswith(".m8")]

    if not m8_files:
        print(f"❌ No .m8 files found for {ticket}!")
        return None

    return m8_files


def extract_max_tm_score(m8_files, ticket):
    """ Finds the highest TM-score across all extracted .m8 files for a given job (ticket). """
    max_tm = 0.0

    for m8_file in m8_files:
        try:
            result = subprocess.run(
                f"awk -F'\t' '{{print $12}}' {m8_file} | sort -nr | head -1",
                shell=True, capture_output=True, text=True
            )
            score = result.stdout.strip()
            if score:
                score = float(score)
                if score > max_tm:
                    max_tm = score
            print(f"🔍 [{ticket}] {m8_file} → TM-score: {score}")
        except Exception as e:
            print(f"❌ Error extracting TM-score from {m8_file} for job {ticket}: {e}")

    print(f"⭐ [{ticket}] Highest TM-score found: {max_tm}")
    return max_tm


def calculate_tm_scores(num_sequences=10):
    """ Runs Foldseek on all PDBs in parallel and extracts max TM-score. """
    pdb_files = [os.path.join(PDB_FOLDER, f"generated_protein_{i}.pdb") for i in range(num_sequences)]

    tickets = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(submit_to_foldseek, pdb): pdb for pdb in pdb_files}
        for future in concurrent.futures.as_completed(futures):
            pdb_file = futures[future]
            ticket = future.result()
            if ticket:
                tickets[ticket] = pdb_file

    if not tickets:
        print("❌ No jobs were successfully submitted!")
        return None

    completed_tickets = wait_for_completion(set(tickets.keys()))

    if not completed_tickets:
        print("❌ No jobs completed successfully!")
        return None

    all_m8_files = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(download_results, ticket, M8_FOLDER): ticket for ticket in completed_tickets}
        for future in concurrent.futures.as_completed(futures):
            ticket = futures[future]
            m8_files = future.result()
            if m8_files:
                all_m8_files[ticket] = m8_files  # Store m8 files grouped by ticket

    if not all_m8_files:
        print("❌ No .m8 files found!")
        return None

    max_tm_scores = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(extract_max_tm_score, m8_files, ticket): ticket for ticket, m8_files in all_m8_files.items()}
        for future in concurrent.futures.as_completed(futures):
            score = future.result()
            if score is not None:
                max_tm_scores.append(score)

    if not max_tm_scores:
        print("❌ No TM-scores extracted!")
        return None

    avg_tm_score = sum(max_tm_scores) / len(max_tm_scores)
    print(f"📊 All Max TM-Scores: {max_tm_scores}")
    print(f"FINAL AVERAGE MAX TM-SCORE: {avg_tm_score}")
    return avg_tm_score