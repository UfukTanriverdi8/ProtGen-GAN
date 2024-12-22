import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_checkpoint_path = f"../results/40epoch_checkpoint"

tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




def load_and_tokenize_dataset(tokenizer, train_file, val_file=None, max_length=512):
    """
    Load and tokenize the dataset.
    """
    if val_file:
        datasets = load_dataset("text", data_files={
            "train": train_file,
            "validation": val_file
        })
    else:
        print("No validation file provided. Only training data will be loaded.")
        datasets = load_dataset("text", data_files={
            "train": train_file,
        })

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

    tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_datasets

class DNMTDataset(Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data["input_ids"]
        self.attention_mask = tokenized_data["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.float),
        }

def create_dataloaders(tokenized_datasets, batch_size):
    """
    Create DataLoaders for training and optionally validation.
    """
    # Always create a DataLoader for the training dataset
    train_dataset = DNMTDataset(tokenized_datasets["train"])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Create a DataLoader for the validation dataset only if it exists
    validation_dataloader = None
    if "validation" in tokenized_datasets:
        validation_dataset = DNMTDataset(tokenized_datasets["validation"])
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    return train_dataloader, validation_dataloader

# load and tokenize the whole dataset
tokenized_datasets = load_and_tokenize_dataset(
    tokenizer,
    train_file="../data/dnmt_data/dnmt_full.txt",
    val_file="data/dnmt_val.txt",
    max_length=512
)

# get dataloaders
batch_size = 4
train_dataloader, validation_dataloader = create_dataloaders(tokenized_datasets, batch_size)

class Generator(nn.Module):
    def __init__(self, protbert_model):
        super().__init__()
        self.protbert = protbert_model

    def forward(self, input_ids, attention_mask=None):
        outputs = self.protbert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
    def generate(self, input_ids, attention_mask=None, temperature=1.0):
        outputs = self.protbert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        logits = logits / temperature
        probabilities = F.softmax(logits, dim=-1)
        predicted_ids = torch.multinomial(probabilities.view(-1, probabilities.size(-1)), num_samples=1)
        predicted_ids = predicted_ids.view(probabilities.size(0), probabilities.size(1))

        return predicted_ids

class Critic(nn.Module):
    def __init__(self, protbert_model):
        super().__init__()
        self.protbert = protbert_model
        hidden_size = self.protbert.config.hidden_size  # 1024

        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size // 16),
            nn.ReLU(),
            nn.Linear(hidden_size // 16, 1)
        )

    def forward(self, input_data, attention_mask=None):
        if input_data.dim() == 2:  # Token IDs
            outputs = self.protbert(
                input_ids=input_data,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden_state = outputs.hidden_states[-1]
        elif input_data.dim() == 3:  # Embeddings
            last_hidden_state = input_data

        cls_output = last_hidden_state[:, 0, :]  # CLS token embedding
        logits = self.classifier(cls_output)
        return logits



def critic_loss(real_scores, fake_scores, gradient_penalty, lambda_gp=10):
    return -torch.mean(real_scores) + torch.mean(fake_scores) + lambda_gp * gradient_penalty

def generator_loss(fake_scores):
    return -torch.mean(fake_scores)


def compute_gradient_penalty(critic, real_data, fake_data, device):
    # Interpolate embeddings instead of token IDs
    alpha = torch.rand(real_data.size(0), 1, 1).to(device)  # Random interpolation factor
    real_embeds = critic.protbert.bert.embeddings(real_data)  # Get embeddings for real data
    fake_embeds = critic.protbert.bert.embeddings(fake_data)  # Get embeddings for fake data
    interpolates = (alpha * real_embeds + (1 - alpha) * fake_embeds).requires_grad_(True)  # Interpolate
    attention_mask = torch.ones(interpolates.size()[:2]).to(device)  # Full mask (no padding)

    # Pass interpolated embeddings through the critic
    critic_scores = critic(
        interpolates,
        attention_mask=attention_mask,
    )

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=critic_scores,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)  # Flatten
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()  # Gradient penalty
    return gradient_penalty


epoch_to_load = 3

gen_dir = f"checkpoints/Epoch_{epoch_to_load}/generator_bert"
critic_bert_dir = f"checkpoints/Epoch_{epoch_to_load}/critic_bert"
critic_classifier_path = f"checkpoints/Epoch_{epoch_to_load}/critic_classifier.pth"


generator_protbert = AutoModelForMaskedLM.from_pretrained(model_checkpoint_path).to(device)
critic_protbert = AutoModelForMaskedLM.from_pretrained(model_checkpoint_path).to(device)

generator = Generator(protbert_model=generator_protbert).to(device)
critic = Critic(protbert_model=critic_protbert).to(device)

# Optimizers
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))

# Training parameters
n_epochs = 100
n_critic = 5  # Number of critic updates per generator update
lambda_gp = 10  # Gradient penalty weight
seq_length = 512

# wandb
wandb.init(project="ProtGen GAN Training", name="demo-1", config={
        "epochs": n_epochs,
        "critic_updates_per_gen": n_critic,
        "seq_length": seq_length,
        "lambda_gp": lambda_gp,
        "learning_rate": 1e-4,
    })

# Training loop
for epoch in range(n_epochs):
    print(f"Epoch {epoch + 1}/{n_epochs}")
    train_pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{n_epochs}")
    batch_number = 0

    for batch in train_pbar:
        real_data = batch["input_ids"].to(device)
        attention_mask_real = batch["attention_mask"].to(device)
        
        # Generate fake data
        noise = torch.full((real_data.size(0), seq_length), tokenizer.mask_token_id).to(device)
        fake_data = generator.generate(noise)

        # ---------------------
        # Train Critic
        # ---------------------
        for _ in range(n_critic):
            critic_optimizer.zero_grad()

            # Get scores
            real_scores = critic(real_data, attention_mask=attention_mask_real)
            fake_scores = critic(fake_data)

            # Compute gradient penalty
            gp = compute_gradient_penalty(critic, real_data, fake_data, device)

            # Critic loss
            c_loss = critic_loss(real_scores, fake_scores, gp, lambda_gp)
            c_loss.backward()
            critic_optimizer.step()

        # ---------------------
        # Train Generator
        # ---------------------
        gen_optimizer.zero_grad()

        # Generate new fake data
        fake_data = generator.generate(noise)
        fake_scores = critic(fake_data)

        # Generator loss
        g_loss = generator_loss(fake_scores)
        g_loss.backward()
        gen_optimizer.step()

        if batch_number % 10 == 0:
            wandb_log = {
            "Epoch": epoch + 1,
            "Critic Loss": c_loss.item(),
            "Generator Loss": g_loss.item(),
            "Gradient Penalty": gp.item(),
            "Real Score Mean": real_scores.mean().item(),
            "Fake Score Mean": fake_scores.mean().item(),
            }
            wandb.log(wandb_log)
            print(f"Epoch {epoch + 1}/{n_epochs} - Batch {batch_number}")
            print(f"Critic Loss: {c_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}")
            samples = generator.generate(noise)
            generated_sequence = tokenizer.convert_ids_to_tokens(samples[0])
            print(" ".join(generated_sequence))


        batch_number += 1
        

    print(f"Epoch {epoch + 1}/{n_epochs} - Critic Loss: {c_loss.item():.4f} - Generator Loss: {g_loss.item():.4f}")

    # Directory for the current epoch
    epoch_dir = f"checkpoints/epoch_{epoch + 1}"
    os.makedirs(epoch_dir, exist_ok=True)

    # Save critic protbert
    critic_bert_dir = f"{epoch_dir}/critic_bert"
    critic.protbert.save_pretrained(critic_bert_dir)

    # Save the classifier of critic
    critic_classifier_path = f"{epoch_dir}/critic_classifier.pth"
    torch.save(critic.classifier.state_dict(), critic_classifier_path)

    # Save the generator
    gen_dir = f"{epoch_dir}/generator_bert"
    generator.protbert.save_pretrained(gen_dir)

    print(f"Models saved for epoch {epoch + 1}:")
    print(f" - Critic ProtBERT saved at: {critic_bert_dir}")
    print(f" - Critic Classifier saved at: {critic_classifier_path}")
    print(f" - Generator ProtBERT saved at: {gen_dir}")




