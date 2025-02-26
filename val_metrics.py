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

def clean_m8_folder():
    """ Deletes all files and subdirectories inside validation/m8s/. """
    m8_folder = "validation/m8s/"
    
    if os.path.exists(m8_folder):
        shutil.rmtree(m8_folder)  # Deletes the entire directory
        os.makedirs(m8_folder, exist_ok=True)  # Recreate an empty folder
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

def calculate_plddt_scores_and_save_pdb(generator, esm_tokenizer, folding_model, tokenizer, num_sequences=10, batch_size=4, device="cuda"):
    generated_sequences = []

    # Step 1: Generate `num_sequences` from the generator
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
        #print("="*20)
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
            #print(generated_ids)
            #decoded_ids = tokenizer.decode(generated_ids[0])
            #print(f"Mask rate: {generated_ids[0].tolist().count(4)}/{sample_length}")
            #print("*"*20)

        # Decode the generated sequence and clean it
        decoded_sequence = tokenizer.decode(fully_masked_input[0], skip_special_tokens=True).replace(" ", "")[:600]
        generated_sequences.append(decoded_sequence)
    print("*"*10 + " Generated Sequences " + "*"*10)
    print(generated_sequences)
    print("📊 Sequence lengths:",[len(seq) for seq in generated_sequences])
    # Step 2: Process sequences in batches for ESMFold
    plddt_scores = []
    os.makedirs("./validation/pdbs", exist_ok=True)

    for batch_start in range(0, len(generated_sequences), batch_size):
        batch_sequences = generated_sequences[batch_start:batch_start + batch_size]

        # Tokenize the batch for ESMFold
        tokenized_inputs = esm_tokenizer(batch_sequences, return_tensors="pt", add_special_tokens=False, padding=True)['input_ids'].to(device)

        with torch.no_grad():
            outputs = folding_model(tokenized_inputs)

        # Step 3: Convert outputs to PDB using the existing function
        pdb_list = convert_outputs_to_pdb(outputs)

        # Step 4: Save each PDB file and calculate the pLDDT score
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
    """ Runs Foldseek on all 10 PDBs in parallel and extracts max TM-score. """
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


#calculate_tm_scores()