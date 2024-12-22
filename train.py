import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from models import Generator, Critic
import os
from tqdm import tqdm
import wandb
from loss import critic_loss, generator_loss, compute_gradient_penalty
from dataset import load_and_tokenize_dataset, get_dataloaders
from metrics import calculate_sequence_identity


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "True"

model_checkpoint_path = f"../checkpoints/dynamic-masked/checkpoint-7610379"

tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenized_datasets = load_and_tokenize_dataset(
    tokenizer,
    gen_file = "data/dnmt_gen.txt",
    critic_file = "data/dnmt_critic.txt",
    max_length=512
)

batch_size = 4
gen_dataloader, critic_dataloader = get_dataloaders(tokenized_datasets, batch_size)

generator_protbert = AutoModelForMaskedLM.from_pretrained(model_checkpoint_path).to(device)
critic_protbert = AutoModelForMaskedLM.from_pretrained(model_checkpoint_path).to(device)

generator = Generator(protbert_model=generator_protbert).to(device)
critic = Critic(protbert_model=critic_protbert).to(device)

gen_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))



# wandb
""" wandb.init(project="ProtGen GAN Training", name="demo-1", config={
        "epochs": n_epochs,
        "critic_updates_per_gen": n_critic,
        "lambda_gp": lambda_gp,
        "learning_rate": 1e-4,
    }) """

# Params
n_epochs = 100
n_critic = 5  # Number of critic updates per generator update
lambda_gp = 10  # Gradient penalty weight
initial_masking_rate = 0.9

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

            # Initialize masking rate for this sequence
            current_masking_rate = initial_masking_rate

            # Iterative filling
            while current_masking_rate > 0.1:
                # Apply dynamic masking
                masked_input_ids = input_ids.clone()
                seq_length_per_batch = attention_mask.sum(dim=1)  # sequence that does not include pad or masks
                for i in range(len(masked_input_ids)):
                    num_to_mask = int(seq_length_per_batch[i].item() * current_masking_rate) # how many items will be masked
                    mask_indices = torch.randperm(seq_length_per_batch[i].item())[:num_to_mask] # 
                    masked_input_ids[i, mask_indices] = tokenizer.mask_token_id

                # Generate fake data
                fake_data = generator(masked_input_ids, attention_mask)

                # ---------------------
                # Train Critic
                # ---------------------
                for _ in range(n_critic):
                    try:
                        # Critic batch
                        critic_batch = next(critic_iter)
                        real_data = critic_batch["input_ids"].to(device)
                        attention_mask_real = critic_batch["attention_mask"].to(device)
                    except StopIteration:
                        critic_iter = iter(critic_dataloader)
                        critic_batch = next(critic_iter)
                        real_data = critic_batch["input_ids"].to(device)
                        attention_mask_real = critic_batch["attention_mask"].to(device)

                    critic_optimizer.zero_grad()

                    # Get scores
                    real_scores = critic(real_data, attention_mask=attention_mask_real)
                    fake_scores = critic(fake_data, attention_mask=attention_mask)

                    # Compute gradient penalty
                    gp = compute_gradient_penalty(critic, real_data, fake_data, device)

                    # Critic loss
                    c_loss = -torch.mean(real_scores) + torch.mean(fake_scores) + lambda_gp * gp
                    c_loss.backward()
                    critic_optimizer.step()

                # ---------------------
                # Train Generator
                # ---------------------
                gen_optimizer.zero_grad()
                
                # Critic feedback for generator
                fake_scores = critic(fake_data, attention_mask)

                # Generator loss
                g_loss = -torch.mean(fake_scores)
                g_loss.backward()
                gen_optimizer.step()

                # Decrease masking rate after each step
                current_masking_rate = max(0.1, current_masking_rate - 0.1)

        except StopIteration:
            break

        if batch_number % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs} - Batch {batch_number}")
            print(f"Critic Loss: {c_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}")

        batch_number += 1

    # Save models
    epoch_dir = f"./checkpoints/gan/epoch_{epoch + 1}"
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

