import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, EsmForProteinFolding
from models import Generator, Critic
import os
import wandb
from loss import critic_loss, generator_loss, compute_gradient_penalty
from dataset import load_and_tokenize_dataset, get_dataloaders
from torch.optim import AdamW
from val_metrics import calculate_plddt_scores_and_save_pdb, calculate_tm_scores, clean_m8_folder
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
os.environ["TOKENIZERS_PARALLELISM"] = "true"

model_checkpoint_path = f"../checkpoints/dynamic/saved-final-300"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path, do_lower_case=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenized_datasets = load_and_tokenize_dataset(
    tokenizer,
    gen_file = "data/dnmt_gen.txt",
    critic_file = "data/dnmt_critic.txt",
    max_length = 512,
    fully_masked = False
)

batch_size = 4
gen_dataloader, critic_dataloader = get_dataloaders(tokenized_datasets, batch_size)

generator_protbert = AutoModelForMaskedLM.from_pretrained(model_checkpoint_path).to(device)
critic_protbert = AutoModelForMaskedLM.from_pretrained(model_checkpoint_path).to(device)

generator = Generator(protbert_model=generator_protbert, mask_token_id = tokenizer.mask_token_id).to(device)
critic = Critic(protbert_model=critic_protbert).to(device)


gen_optimizer = AdamW(generator.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=0.01)
critic_optimizer = AdamW(critic.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=0.01)

# wandb
run_name = "10percent_run"
wandb.init(project="ProtGen GAN Training", name=run_name, mode="online")

# Params
n_epochs = 10
n_critic = 8  # Number of critic updates per generator update
lambda_gp = 5  # Gradient penalty weight
initial_masking_rate = 0.9
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

            min_temp = 0.8
            max_temp = 1.2
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
                    print(f"Batch element {i}, Mask count {sample_mask_count}, Mask rate: {sample_mask_count / meaningful_seq[i].sum().item():.2f}")
            
                current_masking_rate = max(0, current_masking_rate - iteration_fill_rate)
                if (final_input_ids == tokenizer.mask_token_id).sum() == 0:
                    print("No [MASK] tokens remain, exiting in the loop!")
                    break


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
            attention_mask_fake = (fake_data != tokenizer.pad_token_id).long()
            fake_scores = critic(fake_data, attention_mask=attention_mask_fake)
            g_loss = generator_loss(fake_scores)
            # gradient clipping +5 ile -5 ya da +1 -1 ile
            # bu da olmazsa gradient penalty ekle
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5)

            gen_optimizer.step()
            real_score_mean = real_scores.mean().item()
            fake_score_mean = fake_scores.mean().item()

        except StopIteration:
            break

        if batch_number % 630 == 0: 
            print("=" * 20)
            print(f"Epoch {epoch + 1}/{n_epochs} - Batch {batch_number}")
            print(f"Critic Loss: {c_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}")
            print(f"Critic Real Score Mean: {real_score_mean:.4f}, Fake Score Mean: {fake_score_mean:.4f}")
            avg_plddt_score = calculate_plddt_scores_and_save_pdb(generator, esmfold_tokenizer, esmfold_model, tokenizer,batch_size=4, num_sequences=10)
            print(f"Average pLDDT score: {avg_plddt_score:.2f}")
            avg_max_tm_score = calculate_tm_scores(num_sequences=10)
            print(f"Average max TM-Score: {avg_max_tm_score}")

            wandb.log({
                "epoch": epoch + 1,
                "batch": batch_number + 1,
                "critic_loss": c_loss.item(),
                "generator_loss": g_loss.item(),
                "plddt_score": avg_plddt_score,
                "max_tm_score": avg_max_tm_score,
            })
            clean_m8_folder()

        batch_number += 1
        

    print("=" * 30)
    print(f"End Of Epoch {epoch + 1}/{n_epochs} - Batch {batch_number}")
    print(f"Critic Loss: {c_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}")
    print(f"Critic Real Score Mean: {real_score_mean:.4f}, Fake Score Mean: {fake_score_mean:.4f}")
    avg_plddt_score = calculate_plddt_scores_and_save_pdb(generator, esmfold_tokenizer, esmfold_model, tokenizer,batch_size=4, num_sequences=10)
    print(f"Average pLDDT score: {avg_plddt_score:.2f}")
    avg_max_tm_score = calculate_tm_scores(num_sequences=10)
    print(f"Average max TM-Score: {avg_max_tm_score}")
    print("=" * 30)

    wandb.log({
        "epoch": epoch + 1,
        "batch": batch_number,
        "critic_loss": c_loss.item(),
        "generator_loss": g_loss.item(),
        "plddt_score": avg_plddt_score,
        "max_tm_score": avg_max_tm_score,
    })
    

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
