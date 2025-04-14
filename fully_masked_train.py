import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, EsmForProteinFolding
from models import Generator, Critic
import wandb
from loss import critic_loss, generator_loss, compute_gradient_penalty
from dataset import load_and_tokenize_dataset, get_dataloaders
from torch.optim import AdamW
from val_metrics import *
from torch.nn.utils.rnn import pad_sequence
torch.backends.cuda.matmul.allow_tf32 = True


os.environ["TOKENIZERS_PARALLELISM"] = "true"

model_checkpoint_path = f"../checkpoints/dynamic/saved-final-300"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path, do_lower_case=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenized_datasets = load_and_tokenize_dataset(
    tokenizer,
    gen_file = "data/dnmt_gen.txt",
    critic_file = "data/dnmt_critic.txt",
    max_length = 512,
    fully_masked = True,
    full_dataset="data/dnmt_full.txt"
)

batch_size = 4
critic_dataloader = get_dataloaders(tokenized_datasets, batch_size)

generator_protbert = AutoModelForMaskedLM.from_pretrained(model_checkpoint_path).to(device)
critic_protbert = AutoModelForMaskedLM.from_pretrained(model_checkpoint_path).to(device)

generator = Generator(protbert_model=generator_protbert, mask_token_id = tokenizer.mask_token_id).to(device)
critic = Critic(protbert_model=critic_protbert).to(device)


gen_optimizer = AdamW(generator.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=0.01)
critic_optimizer = AdamW(critic.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=0.01)

# wandb
run_name = "fully_masked_test"
wandb.init(project="ProtGen GAN Training", name=run_name, mode="online")

# Params
n_epochs = 10
n_critic = 8  # Number of critic updates per generator update
lambda_gp = 5  # Gradient penalty weight
initial_masking_rate = 1.0
iteration_fill_rate = 0.1
min_temp = 0.8
max_temp = 1.2

debug_seq = "TIALRPDRLTQVLGTEVPTDEGTRLLGAIGFDVEAGEDALHCTVPTWRPDVSIEEDLIEEVA"


# ESMFold Initialization
esmfold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).to(device)
esmfold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
esmfold_model.esm = esmfold_model.esm.half()
esmfold_model.eval()


def generate_fake_batch(generator, tokenizer, batch_size_local, sample_file,
                        initial_masking_rate, iteration_fill_rate,
                        min_temp, max_temp, max_len, device, debug=True):
    """
    Generate a fake batch by sampling sequence lengths and iteratively filling tokens.
    If debug is True, prints debug info.
    """
    gen_inputs, gen_attn_masks, seq_lengths = [], [], []
    for _ in range(batch_size_local):
        seq_length = sample_sequence_length(sample_file)
        total_length = seq_length + 2  # account for [CLS] and [SEP]
        seq_lengths.append(total_length)
        # Create fully masked sequence with CLS and SEP.
        seq = torch.full((total_length,), tokenizer.mask_token_id, device=device)
        seq[0] = tokenizer.cls_token_id
        seq[-1] = tokenizer.sep_token_id
        gen_inputs.append(seq)
        gen_attn_masks.append(torch.ones(total_length, dtype=torch.long, device=device))
        
    if debug:
        print("Sampled sequence lengths:", seq_lengths)
    final_input_ids = pad_sequence(gen_inputs, batch_first=True, padding_value=tokenizer.pad_token_id)

    iteration_count = 0
    current_masking_rate = initial_masking_rate
    while current_masking_rate > 0:
        iteration_count += 1
        updated_attention_mask = ((final_input_ids != tokenizer.mask_token_id) & 
                                  (final_input_ids != tokenizer.pad_token_id)).long()
        temperature = min_temp + torch.rand(1).item() * (max_temp - min_temp)
        final_input_ids = generator.generate(
            final_input_ids,
            updated_attention_mask,
            temperature=temperature,
            keep_percent=iteration_fill_rate,
            current_rate=current_masking_rate
        )
        if debug:
            print("*" * 50)
            print(f"Filling iteration #{iteration_count}")
            for i in range(final_input_ids.size(0)):
                tokens = final_input_ids[i].tolist()
                sample_tokens = tokenizer.convert_ids_to_tokens(tokens)
                mask_count = sample_tokens.count("[MASK]")
                # Compute number of meaningful tokens based on original input.
                meaningful = ((gen_inputs[i] != tokenizer.pad_token_id) &
                              (gen_inputs[i] != tokenizer.cls_token_id) &
                              (gen_inputs[i] != tokenizer.sep_token_id)).sum().item()
                meaningful = meaningful if meaningful > 0 else 1
                print(f"Batch element {i}, Mask count: {mask_count}, Mask rate: {mask_count/meaningful:.2f}")
        current_masking_rate = max(0, current_masking_rate - iteration_fill_rate)
        if (final_input_ids == tokenizer.mask_token_id).sum() == 0:
            if debug:
                print("No [MASK] tokens remain, exiting iterative filling loop!")
            break

    fake_data = final_input_ids
    # Pad or truncate to a fixed maximum length.
    if fake_data.size(1) < max_len:
        pad_length = max_len - fake_data.size(1)
        fake_data = torch.nn.functional.pad(fake_data, (0, pad_length), value=tokenizer.pad_token_id)
    elif fake_data.size(1) > max_len:
        fake_data = fake_data[:, :max_len]
        
    return fake_data


# Training loop
for epoch in range(n_epochs):
    print(f"Epoch {epoch + 1}/{n_epochs}")
    critic_iter = iter(critic_dataloader)
    batch_number = 0

    while True:
        critic_loss_val = 0.0
        break_epoch = False  # Flag to break when critic dataloader is exhausted

        # (1) Critic Updates: perform n_critic critic steps.
        for _ in range(n_critic):
            try:
                critic_batch = next(critic_iter)
            except StopIteration:
                break_epoch = True
                break
            real_data = critic_batch["input_ids"].to(device)
            attn_mask_real = critic_batch["attention_mask"].to(device)
            critic_optimizer.zero_grad()

            # Generate a fresh fake batch.
            fake_data = generate_fake_batch(
                generator,
                tokenizer,
                batch_size_local=4,
                sample_file="data/dnmt_unformatted.txt",
                initial_masking_rate=initial_masking_rate,
                iteration_fill_rate=iteration_fill_rate,
                min_temp=min_temp,
                max_temp=max_temp,
                max_len=512,
                device=device
            )
            # Detach fake_data for critic updates.
            fake_data_critic = fake_data.detach()

            gradient_penalty = compute_gradient_penalty(critic, real_data, fake_data_critic, device)
            real_scores = critic(real_data, attention_mask=attn_mask_real)
            attn_mask_fake = (fake_data_critic != tokenizer.pad_token_id).long()
            fake_scores = critic(fake_data_critic, attention_mask=attn_mask_fake)
            c_loss = critic_loss(real_scores, fake_scores, gradient_penalty, lambda_gp)
            c_loss.backward()
            critic_optimizer.step()
            critic_loss_val += c_loss.item()
            
        if break_epoch:
            break  # End epoch when the critic dataloader is exhausted.

        critic_loss_val /= n_critic  # Average critic loss.

        # (2) Generator Update: generate a new fake batch (without detachment).
        gen_optimizer.zero_grad()
        fake_data = generate_fake_batch(
            generator,
            tokenizer,
            batch_size_local=4,
            sample_file="data/dnmt_unformatted.txt",
            initial_masking_rate=initial_masking_rate,
            iteration_fill_rate=iteration_fill_rate,
            min_temp=min_temp,
            max_temp=max_temp,
            max_len=512,
            device=device
        )
        attn_mask_fake = (fake_data != tokenizer.pad_token_id).long()
        fake_scores = critic(fake_data, attention_mask=attn_mask_fake)
        g_loss = generator_loss(fake_scores)
        g_loss.backward()
        gen_optimizer.step()

        # Logging and debugging.
        
        if batch_number % 250 == 0:
            print("=" * 20)
            print(f"Epoch {epoch+1}/{n_epochs} - Batch {batch_number}")
            print(f"Critic Loss (avg): {critic_loss_val:.4f}")
            print(f"Generator Loss:    {g_loss.item():.4f}")

            generated_sequences = generate_fake_sequences(
                generator=generator,
                tokenizer=tokenizer,
                num_sequences=10
            )

            avg_plddt_score = calculate_plddt_scores_and_save_pdb(
                generated_sequences,
                folding_model=esmfold_model,
                folding_tokenizer=esmfold_tokenizer,
                batch_size=4,
                num_sequences=10
            )
            print(f"Average pLDDT score: {avg_plddt_score:.2f}")

            avg_max_tm_score = calculate_tm_scores(num_sequences=10)
            print(f"Average max TM-Score: {avg_max_tm_score}")

            avg_scAcc = calculate_mpnn_alignment_metric(
                generated_sequences=generated_sequences,
            )

            avg_progres = compute_average_progres_score()

            avg_pairwise_tm_score = calculate_pairwise_tm_score()

            wandb.log({
                "epoch": epoch+1,
                "batch": batch_number+1,
                "critic_loss": critic_loss_val,
                "generator_loss": g_loss.item(),
                "plddt_score": avg_plddt_score,
                "max_tm_score": avg_max_tm_score,
                "scAccuracy": avg_scAcc,
                "progres": avg_progres,
                "pairwise_tm_score": avg_pairwise_tm_score,
            })
            clean_m8_folder()
        batch_number += 1
    
    ### End of epoch logging and saving a checkpoint

    print("=" * 30)
    print(f"End Of Epoch {epoch+1}/{n_epochs} - Batch {batch_number}")
    print(f"Critic Loss: {c_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}")
    generated_sequences = generate_fake_sequences(
                generator=generator,
                tokenizer=tokenizer,
                num_sequences=10
            )

    avg_plddt_score = calculate_plddt_scores_and_save_pdb(
        generated_sequences,
        folding_model=esmfold_model,
        folding_tokenizer=esmfold_tokenizer,
        batch_size=4,
        num_sequences=10
    )
    print(f"Average pLDDT score: {avg_plddt_score:.2f}")

    avg_max_tm_score = calculate_tm_scores(num_sequences=10)
    print(f"Average max TM-Score: {avg_max_tm_score}")

    avg_scAcc = calculate_mpnn_alignment_metric(
        generated_sequences=generated_sequences,
    )

    avg_progres = compute_average_progres_score()

    avg_pairwise_tm_score = calculate_pairwise_tm_score()

    wandb.log({
        "epoch": epoch+1,
        "batch": batch_number,
        "critic_loss": c_loss.item(),
        "generator_loss": g_loss.item(),
        "plddt_score": avg_plddt_score,
        "max_tm_score": avg_max_tm_score,
        "scAccuracy": avg_scAcc,
        "progres": avg_progres,
        "pairwise_tm_score": avg_pairwise_tm_score,
    })

    # --------------------------
    #  Saving the models
    # --------------------------
    save_dir = f"./checkpoints/{run_name}/epoch_{epoch+1}"
    os.makedirs(save_dir, exist_ok=True)

    critic_bert_dir = f"{save_dir}/critic_bert"
    critic.protbert.save_pretrained(critic_bert_dir)

    critic_classifier_path = f"{save_dir}/critic_classifier.pth"
    torch.save(critic.classifier.state_dict(), critic_classifier_path)

    gen_dir = f"{save_dir}/generator_bert"
    generator.protbert.save_pretrained(gen_dir)

    print(f"Models saved for epoch {epoch+1}:")
    print(f" - Critic ProtBERT saved at: {critic_bert_dir}")
    print(f" - Critic Classifier saved at: {critic_classifier_path}")
    print(f" - Generator ProtBERT saved at: {gen_dir}")
