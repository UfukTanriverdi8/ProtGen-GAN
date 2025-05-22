import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM, EsmForProteinFolding
from models import Generator, Critic
import wandb
from loss import critic_loss, generator_loss, compute_gradient_penalty
from dataset import load_and_tokenize_dataset, get_dataloaders, get_dynamic_dataloaders
from torch.optim import AdamW
from val_metrics import *
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# -----------------------
# Argument parsing
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="ProtGen GAN Training - 10P Mode")
    parser.add_argument("--n_critic",   type=int,   default=8,
                        help="Number of critic updates per generator update")
    parser.add_argument("--lambda_gp",  type=float, default=5.0,
                        help="Gradient penalty weight")
    parser.add_argument("--lr_gen",     type=float, default=5e-5,
                        help="Learning rate for generator optimizer")
    parser.add_argument("--lr_critic",  type=float, default=5e-5,
                        help="Learning rate for critic optimizer")
    parser.add_argument("--wd_gen",     type=float, default=0.01,
                        help="Weight decay for generator optimizer")
    parser.add_argument("--wd_critic",  type=float, default=0.01,
                        help="Weight decay for critic optimizer")
    parser.add_argument("--n_epochs",   type=int,   default=10,
                        help="Number of training epochs")
    parser.add_argument("--run_name",   type=str,   default="10percent_with_metrics",
                        help="W&B run name"),
    parser.add_argument("--batch_size", type=int,   default=4,
                        help="Batch size for training")
    return parser.parse_args()

args = parse_args()

model_checkpoint_path = "../checkpoints/dynamic/saved-final-300"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path, do_lower_case=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenized_full_dataset = load_and_tokenize_dataset(
    tokenizer,
    full_dataset="data/dnmt_full.txt",
    fully_masked=True,
    max_length=512
)["critic"]


generator_protbert = AutoModelForMaskedLM.from_pretrained(model_checkpoint_path).to(device)
critic_protbert    = AutoModelForMaskedLM.from_pretrained(model_checkpoint_path).to(device)

generator = Generator(protbert_model=generator_protbert, mask_token_id=tokenizer.mask_token_id).to(device)
critic    = Critic(protbert_model=critic_protbert).to(device)

gen_optimizer    = AdamW(generator.parameters(), lr=args.lr_gen, betas=(0.9, 0.999), weight_decay=args.wd_gen)
critic_optimizer = AdamW(critic.parameters(),   lr=args.lr_critic, betas=(0.9, 0.999), weight_decay=args.wd_critic)

wandb.init(project="ProtGen GAN Training", name=args.run_name, mode="disabled")

# --------------------------
# Hyperparameters / Settings
# --------------------------
n_epochs            = args.n_epochs
n_critic            = args.n_critic
lambda_gp           = args.lambda_gp
initial_masking_rate= 0.9
iteration_fill_rate = 0.1
min_temp            = 0.8
max_temp            = 1.2

# ESMFold Initialization
esmfold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).to(device)
esmfold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
esmfold_model.esm = esmfold_model.esm.half()
esmfold_model.eval()


def generate_fakes_for_batch(generator, tokenizer, input_ids, attention_mask,
                             initial_mask_rate, iteration_rate,
                             min_temp, max_temp):
    device = input_ids.device
    
    meaningful_seq = (
        (input_ids != tokenizer.pad_token_id) &
        (input_ids != tokenizer.unk_token_id) &
        (input_ids != tokenizer.cls_token_id) &
        (input_ids != tokenizer.sep_token_id)
    )
    random_mask = torch.rand(input_ids.shape, device=device) < initial_mask_rate
    mask_indices = meaningful_seq & random_mask

    final_input_ids = input_ids.clone()
    final_input_ids[mask_indices] = tokenizer.mask_token_id

    current_mask_rate = initial_mask_rate
    iteration_count   = 0

    while current_mask_rate > 0:
        iteration_count += 1
        updated_attention_mask = (final_input_ids != tokenizer.pad_token_id).long()

        temperature = min_temp + torch.rand(1).item() * (max_temp - min_temp)
        final_input_ids = generator.generate(
            final_input_ids,
            updated_attention_mask,
            temperature=temperature,
            keep_percent=iteration_rate,
            current_rate=current_mask_rate
        )

        print("*" * 50)
        print(f"Filling iteration #{iteration_count}")
        for i in range(final_input_ids.size(0)):
            tokens = final_input_ids[i].tolist()
            sample = tokenizer.convert_ids_to_tokens(tokens)
            mask_count = sample.count("[MASK]")
            total_meaningful = meaningful_seq[i].sum().item()
            mask_rate = mask_count / total_meaningful if total_meaningful > 0 else 0
            print(f"  Batch element {i}, remaining [MASK]: {mask_count}, mask rate: {mask_rate:.2f}")

        current_mask_rate = max(0, current_mask_rate - iteration_rate)
        if (final_input_ids == tokenizer.mask_token_id).sum() == 0:
            print("No [MASK] tokens remain; exiting fill loop.")
            break

    return final_input_ids



for epoch in range(n_epochs):
    print(f"Epoch {epoch+1}/{n_epochs}")
    gen_dataloader, critic_dataloader = get_dynamic_dataloaders(
        tokenized_full_dataset,
        batch_size=args.batch_size,
        n_critic=args.n_critic
    )



    gen_iter    = iter(gen_dataloader)
    critic_iter = iter(critic_dataloader)
    batch_number = 0


    while True:
        # We'll accumulate critic losses over the n_critic steps
        critic_loss_val = 0.0
        break_epoch = False  # Flag to break out of the while-loop if we exhaust data

        # --------------------------
        #   (1) Train the Critic
        #       n_critic times
        # --------------------------
        for _ in range(n_critic):
            # ----- Fetch new real batch -----
            try:
                real_batch = next(critic_iter)
            except StopIteration:
                critic_iter = iter(critic_dataloader)
                real_batch = next(critic_iter)

            real_data      = real_batch["input_ids"].to(device)
            attn_mask_real = real_batch["attention_mask"].to(device)

            # ----- Fetch new gen batch for fresh fakes -----
            try:
                gen_batch = next(gen_iter)
            except StopIteration:
                print("Generator dataloader exhausted, ending this epoch.")
                break_epoch = True
                break

            input_ids_gen = gen_batch["input_ids"].to(device)
            attn_mask_gen = gen_batch["attention_mask"].to(device)

            # ----- Generate brand-new fake batch -----
            fake_data = generate_fakes_for_batch(
                generator,
                tokenizer,
                input_ids_gen,
                attn_mask_gen,
                initial_masking_rate,
                iteration_fill_rate,
                min_temp,
                max_temp
            )
            fake_data = fake_data.detach()  # Detach to avoid backprop through generator

            # ----- Critic forward/backward -----
            critic_optimizer.zero_grad()

            gradient_penalty = compute_gradient_penalty(
                critic, real_data, fake_data, device
            )
            real_scores = critic(real_data, attention_mask=attn_mask_real)
            fake_scores = critic(
                fake_data, attention_mask=(fake_data != tokenizer.pad_token_id).long()
            )
            c_loss = critic_loss(real_scores, fake_scores, gradient_penalty, lambda_gp)

            c_loss.backward()
            critic_optimizer.step()
            critic_loss_val += c_loss.item()

        # If we broke out early (no more data), end the epoch
        if break_epoch:
            break

        # Take mean critic loss over all n_critic steps
        critic_loss_val /= n_critic

        # --------------------------
        #   (2) Train the Generator
        #       once after n_critic
        # --------------------------
        try:
            gen_batch = next(gen_iter)
        except StopIteration:
            print("Generator dataloader exhausted before generator step.")
            break

        input_ids_gen = gen_batch["input_ids"].to(device)
        attn_mask_gen = gen_batch["attention_mask"].to(device)

        # Generate fresh fakes for the generator update
        fake_data = generate_fakes_for_batch(
            generator,
            tokenizer,
            input_ids_gen,
            attn_mask_gen,
            initial_masking_rate,
            iteration_fill_rate,
            min_temp,
            max_temp
        )

        gen_optimizer.zero_grad()
        fake_scores = critic(
            fake_data, attention_mask=(fake_data != tokenizer.pad_token_id).long()
        )
        g_loss = generator_loss(fake_scores)

        g_loss.backward()
        gen_optimizer.step()


        # -------------- Logging --------------
        if batch_number % 250 == 0 and batch_number > 0:
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
        # ---------------------------------------------

        batch_number += 1
        #break
    # End of epoch logging
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
    save_dir = f"./checkpoints/{args.run_name}/epoch_{epoch+1}"
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
