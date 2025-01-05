import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from models import Generator, Critic
import os
from tqdm import tqdm
import wandb
from loss import critic_loss, generator_loss, compute_gradient_penalty
from dataset import load_and_tokenize_dataset, get_dataloaders
from metrics import calculate_sequence_identity
#torch.manual_seed(4)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "True"

model_checkpoint_path = f"../checkpoints/dynamic-masked/checkpoint-epoch-240"

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
wandb.init(project="ProtGen GAN Training", name="demo-3")

# Params
n_epochs = 100
n_critic = 8  # Number of critic updates per generator update
lambda_gp = 10  # Gradient penalty weight
initial_masking_rate = 0.9
iteration_fill_rate = 0.1

debug_seq = "TIALRPDRLTQVLGTEVPTDEGTRLLGAIGFDVEAGEDALHCTVPTWRPDVSIEEDLIEEVA"

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
            #debug_tokens = list(debug_seq)  # ["K", "T", "I", "A", "L", ...]

            # Add CLS and SEP tokens
            # debug_tokens = ["[CLS]"] + debug_tokens + ["[SEP]"]
            # debug_input_ids = tokenizer.convert_tokens_to_ids(debug_tokens)
            input_ids = gen_batch["input_ids"].to(device)
            #debug_attention_mask = (input_ids != tokenizer.pad_token_id).long()

            attention_mask = gen_batch["attention_mask"].to(device)
            final_input_ids = input_ids.clone()
            """ print("ORIGINAL SEQUENCE")
            for i in range(batch_size):
                print("="*50)
                tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
                print(tokens)
                print(attention_mask[i]) """
                
            # Initial masking before iterative loop
            # seq that does not contain pad, unk, 
            meaningful_seq = (input_ids != tokenizer.pad_token_id) & (input_ids != tokenizer.unk_token_id) & (input_ids != tokenizer.cls_token_id) & (input_ids != tokenizer.sep_token_id)

            # Compute masking indices within meaningful tokens
            random_mask = torch.rand(input_ids.shape, device=device) < initial_masking_rate
            mask_indices = meaningful_seq & random_mask

            # Apply masking
            final_input_ids = input_ids.clone()
            final_input_ids[mask_indices] = tokenizer.mask_token_id

            """ print("BEFORE FILLING")
            for i in range(batch_size):
                print("="*50)
                tokens = tokenizer.convert_ids_to_tokens(final_input_ids[i])
                print(tokens)
                print(f"Mask count: {tokens.count('[MASK]')}")
                #print(attention_mask[i]) """

            iteration_count = 0
            current_masking_rate = initial_masking_rate
            """
            PAD: 0
            UNK: 1
            CLS: 2
            SEP: 3
            MASK: 4
            """

            min_temp = 0.7
            max_temp = 1.5
            # Iterative filling
            while current_masking_rate > 0:
                iteration_count += 1
                updated_attention_mask = ((final_input_ids != tokenizer.mask_token_id) & (final_input_ids != tokenizer.pad_token_id)).long()
                random_temperature = min_temp + torch.rand(1).item() * (max_temp - min_temp)
                generated_ids = generator.generate(final_input_ids, updated_attention_mask, temperature=random_temperature, keep_percent=iteration_fill_rate, current_rate=current_masking_rate)
                final_input_ids = generated_ids

                print("*"*50)
                print(iteration_count)

                for i in range(final_input_ids.size(0)):
                    tokens = final_input_ids[i].tolist() 
                    sample = tokenizer.convert_ids_to_tokens(tokens)
                    sample_mask_count = sample.count("[MASK]")
                    print(f"Batch element {i}, Mask count {sample_mask_count}, Mask rate: {sample_mask_count/len(meaningful_seq[i]):.2f}")

            
                current_masking_rate = max(0, current_masking_rate - iteration_fill_rate)

            # Final fake data generation complete
            fake_data = final_input_ids

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

                # Compute gradient penalty
                gradient_penalty = compute_gradient_penalty(
                    critic, real_data, fake_data, device
                )

                # getting the scores
                real_scores = critic(real_data, attention_mask=attention_mask_real)
                attention_mask_fake = (fake_data != tokenizer.pad_token_id).long()
                fake_scores = critic(fake_data, attention_mask=attention_mask_fake)
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
