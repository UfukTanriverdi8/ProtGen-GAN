import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from models import Generator, Critic
import os
from tqdm import tqdm
import wandb
from loss import critic_loss, generator_loss, compute_gradient_penalty
from dataset import load_and_tokenize_dataset, get_dataloaders
from metrics import calculate_sequence_identity


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#os.environ["WANDB_DISABLED"] = "True"

model_checkpoint_path = f"../checkpoints/dynamic-masked/checkpoint-epoch-123"

tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.tensor([0], device=device)

tokenized_datasets = load_and_tokenize_dataset(
    tokenizer,
    gen_file = "data/dnmt_gen.txt",
    critic_file = "data/dnmt_critic.txt",
    max_length = 512
)

batch_size = 4
gen_dataloader, critic_dataloader = get_dataloaders(tokenized_datasets, batch_size)

generator_protbert = AutoModelForMaskedLM.from_pretrained(model_checkpoint_path).to(device)
critic_protbert = AutoModelForMaskedLM.from_pretrained(model_checkpoint_path).to(device)

generator = Generator(protbert_model=generator_protbert, mask_token_id = tokenizer.mask_token_id).to(device)
critic = Critic(protbert_model=critic_protbert).to(device)

gen_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))



# wandb
wandb.init(project="ProtGen GAN Training", name="demo-2")

# Params
n_epochs = 100
n_critic = 5  # Number of critic updates per generator update
lambda_gp = 10  # Gradient penalty weight
initial_masking_rate = 0.9
iteration_fill_rate = 0.1

# Training loop
for epoch in range(n_epochs):
    print(f"Epoch {epoch + 1}/{n_epochs}")
    gen_iter = iter(gen_dataloader)
    critic_iter = iter(critic_dataloader)
    batch_number = 0

    while True:
        try:
            # generator batch loading
            gen_batch = next(gen_iter)
            input_ids = gen_batch["input_ids"].to(device)
            attention_mask = gen_batch["attention_mask"].to(device)

            # keeping track of the masking rate
            current_masking_rate = initial_masking_rate

            # Iterative filling
            while current_masking_rate > 0.1:
                # Apply dynamic masking
                masked_input_ids = input_ids.clone()
                updated_attention_mask = attention_mask.clone()  # Clone to avoid modifying the original

                seq_length_per_batch = attention_mask.sum(dim=1)  # sequence that does not include pad or masks
                for i in range(len(masked_input_ids)):
                    num_to_mask = int(seq_length_per_batch[i].item() * current_masking_rate) # how many items will be masked
                    mask_indices = torch.randperm(int(seq_length_per_batch[i].item()))[:num_to_mask] # masked indices
                    masked_input_ids[i, mask_indices] = tokenizer.mask_token_id

                    # Update attention mask to exclude newly masked tokens
                    updated_attention_mask[i, mask_indices] = 0

                # Generate fake data
                fake_data = generator.generate(masked_input_ids, updated_attention_mask, keep_percent=iteration_fill_rate)

                # ---------------------
                # Train Critic
                # ---------------------
                for _ in range(n_critic):
                    try:
                        # Critic batch
                        critic_batch = next(critic_iter)
                        real_data = critic_batch["input_ids"].to(device)
                        attention_mask_real = critic_batch["attention_mask"].to(device)

                        # Mask real data for the critic
                        masked_real_data = real_data.clone()
                        updated_attention_mask_real = attention_mask_real.clone()

                        seq_length_real = attention_mask_real.sum(dim=1)  # Lengths of sequences without padding
                        for i in range(len(masked_real_data)):
                            num_to_mask_real = int(seq_length_real[i].item() * current_masking_rate)
                            mask_indices_real = torch.randperm(int(seq_length_real[i].item()))[:num_to_mask_real]
                            masked_real_data[i, mask_indices_real] = tokenizer.mask_token_id
                            updated_attention_mask_real[i, mask_indices_real] = 0

                    except StopIteration:
                        critic_iter = iter(critic_dataloader)
                        critic_batch = next(critic_iter)
                        real_data = critic_batch["input_ids"].to(device)
                        attention_mask_real = critic_batch["attention_mask"].to(device)

                        # Mask real data for the critic in fallback
                        masked_real_data = real_data.clone()
                        updated_attention_mask_real = attention_mask_real.clone()

                        seq_length_real = attention_mask_real.sum(dim=1)
                        for i in range(len(masked_real_data)):
                            num_to_mask_real = int(seq_length_real[i].item() * current_masking_rate)
                            mask_indices_real = torch.randperm(int(seq_length_real[i].item()))[:num_to_mask_real]
                            masked_real_data[i, mask_indices_real] = tokenizer.mask_token_id
                            updated_attention_mask_real[i, mask_indices_real] = 0

                    critic_optimizer.zero_grad()

                    # Compute gradient penalty
                    gradient_penalty = compute_gradient_penalty(
                        critic, masked_real_data, fake_data, device
                    )

                    # getting the scores
                    real_scores = critic(masked_real_data, attention_mask=updated_attention_mask_real)
                    fake_scores = critic(fake_data, attention_mask=None)
                    c_loss = critic_loss(real_scores, fake_scores, gradient_penalty, lambda_gp)
                    c_loss.backward()
                    critic_optimizer.step()

                # ---------------------
                # Train Generator
                # ---------------------
                gen_optimizer.zero_grad()

                # Compute generator loss
                fake_scores = critic(fake_data, attention_mask=None)
                g_loss = generator_loss(fake_scores)
                g_loss.backward()
                gen_optimizer.step()
                print(f"Current masking rate is: {current_masking_rate:.2f}")
                print(f"Epoch {epoch + 1}/{n_epochs} - Batch {batch_number}")
                print(f"Critic Loss: {c_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}")

                # Decrease masking rate after each step
                current_masking_rate = max(0.1, current_masking_rate - iteration_fill_rate)

        except StopIteration:
            break

        if batch_number % 1 == 0:
            print("="*20)
            print(f"Epoch {epoch + 1}/{n_epochs} - Batch {batch_number}")
            print(f"Critic Loss: {c_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}")
            wandb.log({
                "epoch": epoch + 1,
                "batch": batch_number + 1,
                "critic_loss": c_loss.item(),
                "generator_loss": g_loss.item()
            })

        batch_number += 1

    # Save models
    epoch_dir = f"./checkpoints/epoch_{epoch + 1}"
    os.makedirs(epoch_dir, exist_ok=True)

    critic_bert_dir = f"{epoch_dir}/critic_bert"
    critic.protbert.save_pretrained(critic_bert_dir)

    critic_classifier_path = f"{epoch_dir}/critic_classifier.pth"
    torch.save(critic.classifier.state_dict(), critic_classifier_path)

    gen_dir = f"{epoch_dir}/generator_bert"
    generator.protbert.save_pretrained(gen_dir)

    print(f"Models saved for epoch {epoch + 1}:")
    print(f" - Critic ProtBERT saved at: {critic_bert_dir}")
    print(f" - Critic Classifier saved at: {critic_classifier_path}")
    print(f" - Generator ProtBERT saved at: {gen_dir}")
