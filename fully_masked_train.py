import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, EsmForProteinFolding
from models import Generator, Critic
import os
import wandb
from loss import critic_loss, generator_loss, compute_gradient_penalty
from dataset import load_and_tokenize_dataset, get_dataloaders
from torch.optim import AdamW
from val_metrics import calculate_plddt_scores_and_save_pdb, sample_sequence_length
from torch.nn.utils.rnn import pad_sequence


torch.backends.cuda.matmul.allow_tf32 = True
#torch.manual_seed(4)

"""
1. Per-sequence top-k (instead of global top-k).
2. Exact masking (strictly 90% per sequence).
3. Try removing WGAN-GP if it remains unstable—use clipping on the critic instead.
4. Hyperparameters (max_grad_norm, λ gp, optimizer betas/lr/weight decay, etc.) can all be tuned.
5. Partial freezing or additional MLM loss to preserve the “protein knowledge” in ProtBERT.
6. Dont forget to also try by loading from the saved model
"""


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

model_checkpoint_path = f"../checkpoints/dynamic/saved-260"

tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
run_name = "fully_masked_with_plddt"
wandb.init(project="ProtGen GAN Training", name=run_name, mode="online")

# Params
n_epochs = 10
n_critic = 8  # Number of critic updates per generator update
lambda_gp = 5  # Gradient penalty weight
initial_masking_rate = 1.0
iteration_fill_rate = 0.1

debug_seq = "TIALRPDRLTQVLGTEVPTDEGTRLLGAIGFDVEAGEDALHCTVPTWRPDVSIEEDLIEEVA"


# ESMFold Initialization
esmfold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).to("cuda")
esmfold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
esmfold_model.esm = esmfold_model.esm.half()
esmfold_model.eval()

# Training loop
for epoch in range(n_epochs):
    print(f"Epoch {epoch + 1}/{n_epochs}")
    critic_iter = iter(critic_dataloader)
    batch_number = 0

    # We'll run until we've processed one full epoch from the critic dataloader.
    for _ in critic_dataloader:
        # -----------------------------
        # Generator: Build a batch on the fly
        # -----------------------------
        batch_size_local = 4  # same as critic dataloader batch size
        gen_inputs = []
        gen_attn_masks = []
        seq_lengths = []
        # For each element in the batch, sample a sequence length and build a fully masked sequence.
        for _ in range(batch_size_local):
            seq_length = sample_sequence_length("data/dnmt_unformatted.txt")  # sampled length (without CLS/SEP)
            total_length = seq_length + 2  # add [CLS] and [SEP]
            seq_lengths.append(total_length)
            seq = torch.full((total_length,), tokenizer.mask_token_id, device=device)
            seq[0] = tokenizer.cls_token_id
            seq[-1] = tokenizer.sep_token_id
            gen_inputs.append(seq)
            attn = torch.ones_like(seq, dtype=torch.long, device=device)
            gen_attn_masks.append(attn)
        print(seq_lengths)
        # Pad sequences to form a batch tensor.
        final_input_ids = pad_sequence(gen_inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
        generator_attention_mask = pad_sequence(gen_attn_masks, batch_first=True, padding_value=0)

        # -----------------------------
        # Iterative Filling: Generate final fake sequences
        # -----------------------------
        iteration_count = 0
        current_masking_rate = initial_masking_rate
        min_temp = 0.8
        max_temp = 1.2

        # In each iteration, a percentage (iteration_fill_rate) of the remaining mask tokens is filled.
        while current_masking_rate > 0:
            iteration_count += 1
            updated_attention_mask = ((final_input_ids != tokenizer.mask_token_id) & 
                                      (final_input_ids != tokenizer.pad_token_id)).long()
            random_temperature = min_temp + torch.rand(1).item() * (max_temp - min_temp)
            # Call generator.generate; note that your generate() method should handle padded batches.
            generated_ids = generator.generate(
                final_input_ids,
                updated_attention_mask,
                temperature=random_temperature,
                keep_percent=iteration_fill_rate,
                current_rate=current_masking_rate
            )
            final_input_ids = generated_ids  # update with generated tokens

            print("*"*50)
            print(iteration_count)

            # printing each element
            for i in range(final_input_ids.size(0)):
                tokens = final_input_ids[i].tolist()
                sample_tokens = tokenizer.convert_ids_to_tokens(tokens)
                mask_count = sample_tokens.count("[MASK]")
                meaningful = ((gen_inputs[i] != tokenizer.pad_token_id) & 
                              (gen_inputs[i] != tokenizer.cls_token_id) & 
                              (gen_inputs[i] != tokenizer.sep_token_id)).sum().item()
                if meaningful == 0:
                    meaningful = 1  # avoid division by zero
                
                print(f"Batch element {i}, Mask count: {mask_count}, Mask rate: {mask_count / meaningful:.2f}")

            current_masking_rate = max(0, current_masking_rate - iteration_fill_rate)
            if (final_input_ids == tokenizer.mask_token_id).sum() == 0:
                print("No [MASK] tokens remain, exiting iterative filling loop!")
                break


        fake_data = final_input_ids  # final generated sequences
        max_len = 512
        if fake_data.size(1) < max_len:
            pad_length = max_len - fake_data.size(1)
            fake_data = torch.nn.functional.pad(fake_data, (0, pad_length), value=tokenizer.pad_token_id)
        elif fake_data.size(1) > max_len:
            fake_data = fake_data[:, :max_len]


        # -----------------------------
        # Train Critic
        # -----------------------------
        for _ in range(n_critic):
            try:
                critic_batch = next(critic_iter)
            except StopIteration:
                critic_iter = iter(critic_dataloader)
                critic_batch = next(critic_iter)
            real_data = critic_batch["input_ids"].to(device)
            attention_mask_real = critic_batch["attention_mask"].to(device)
            critic_optimizer.zero_grad()

            gradient_penalty = compute_gradient_penalty(critic, real_data, fake_data, device)
            real_scores = critic(real_data, attention_mask=attention_mask_real)
            attention_mask_fake = (fake_data != tokenizer.pad_token_id).long()
            fake_scores = critic(fake_data, attention_mask=attention_mask_fake)
            c_loss = critic_loss(real_scores, fake_scores, gradient_penalty, lambda_gp)
            c_loss.backward()
            critic_optimizer.step()

        # -----------------------------
        # Train Generator
        # -----------------------------
        gen_optimizer.zero_grad()
        attention_mask_fake = (fake_data != tokenizer.pad_token_id).long()
        fake_scores = critic(fake_data, attention_mask=attention_mask_fake)
        g_loss = generator_loss(fake_scores)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5)
        gen_optimizer.step()

        # Debug/logging
        real_score_mean = real_scores.mean().item()
        fake_score_mean = fake_scores.mean().item()
        if batch_number % 50 == 0:
            print("=" * 20)
            print(f"Epoch {epoch + 1}/{n_epochs} - Batch {batch_number}")
            print(f"Critic Loss: {c_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}")
            print(f"Critic Real Score Mean: {real_score_mean:.4f}, Fake Score Mean: {fake_score_mean:.4f}")
            avg_plddt_score = calculate_plddt_scores_and_save_pdb(
                generator, esmfold_tokenizer, esmfold_model, tokenizer,
                batch_size=4, num_sequences=10
            )
            print(f"Average pLDDT score: {avg_plddt_score:.2f}")

            wandb.log({
                "epoch": epoch + 1,
                "batch": batch_number + 1,
                "critic_loss": c_loss.item(),
                "generator_loss": g_loss.item(),
                "plddt_score": avg_plddt_score
            })
            torch.cuda.empty_cache()

        batch_number += 1
    # Save models
    save_dir = f"./checkpoints/{run_name}/epoch_{epoch + 1}"
    os.makedirs(save_dir, exist_ok=True)

    critic_bert_dir = f"{save_dir}/critic_bert"
    critic.protbert.save_pretrained(critic_bert_dir)

    critic_classifier_path = f"{save_dir}/critic_classifier.pth"
    torch.save(critic.classifier.state_dict(), critic_classifier_path)

    gen_dir = f"{save_dir}/generator_bert"
    generator.protbert.save_pretrained(gen_dir)

    print(f"Models saved for epoch {epoch + 1}:")
    print(f" - Critic ProtBERT saved at: {critic_bert_dir}")
    print(f" - Critic Classifier saved at: {critic_classifier_path}")
    print(f" - Generator ProtBERT saved at: {gen_dir}")
