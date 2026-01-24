import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, EsmForProteinFolding
from models import Generator, Critic
import wandb
from loss import critic_loss, generator_loss, compute_gradient_penalty
from dataset import load_and_tokenize_dataset, get_dataloaders
from torch.optim import AdamW
from val_metrics import *
from torch.nn.utils.rnn import pad_sequence


MAIN_FOLDER = "/gpfs/projects/etur29/ufuk/"
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# -----------------------
# Argument parsing
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="ProtGen MN5 GAN Training - Full Mode")
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
    parser.add_argument("--n_epochs",   type=int,   default=20,
                        help="Number of training epochs")
    parser.add_argument("--run_name",   type=str,   default="mn5_default_full_run",
                        help="W&B run name"),
    parser.add_argument("--batch_size", type=int,   default=8,
                        help="Batch size for training"),
    parser.add_argument("--num_eval_sequences", type=int, default=10,
                        help="Number of sequences to evaluate during training"),
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Batch size for evaluation during training")
    return parser.parse_args()

args = parse_args()


model_checkpoint_path = os.path.join(MAIN_FOLDER, "dynamic-finetuned-protbert")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path, do_lower_case=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



tokenized_datasets = load_and_tokenize_dataset(
    tokenizer,
    gen_file = "data/dnmt_gen.txt",
    critic_file = "data/dnmt_critic.txt",
    max_length = 512,
    fully_masked = True,
    full_dataset="data/dnmt_full.txt"
)

batch_size = args.batch_size
critic_dataloader = get_dataloaders(tokenized_datasets, batch_size)

generator_protbert = AutoModelForMaskedLM.from_pretrained(model_checkpoint_path).to(device)
critic_protbert = AutoModelForMaskedLM.from_pretrained(model_checkpoint_path).to(device)

generator = Generator(protbert_model=generator_protbert, mask_token_id = tokenizer.mask_token_id).to(device)
critic = Critic(protbert_model=critic_protbert).to(device)

gen_optimizer    = AdamW(generator.parameters(), lr=args.lr_gen, betas=(0.9, 0.999), weight_decay=args.wd_gen)
critic_optimizer = AdamW(critic.parameters(),   lr=args.lr_critic, betas=(0.9, 0.999), weight_decay=args.wd_critic)

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

debug_seq = "TIALRPDRLTQVLGTEVPTDEGTRLLGAIGFDVEAGEDALHCTVPTWRPDVSIEEDLIEEVA"


# --------------------------
# WandB Initialization
# --------------------------
wandb.init(project="ProtGen GAN Training", name=args.run_name, mode="offline")
wandb.config.update({
    "n_critic": args.n_critic,
    "lambda_gp": args.lambda_gp,
    "lr_gen": args.lr_gen,
    "lr_critic": args.lr_critic,
    "wd_gen": args.wd_gen,
    "wd_critic": args.wd_critic,
    "n_epochs": args.n_epochs,
    "batch_size": args.batch_size,
    "num_eval_sequences": args.num_eval_sequences,
    "eval_batch_size": args.eval_batch_size
})


# ESMFold Initialization
esmfold_path = os.path.join(MAIN_FOLDER, "esmfold")
esmfold_model = EsmForProteinFolding.from_pretrained(esmfold_path, low_cpu_mem_usage=True).to(device)
esmfold_tokenizer = AutoTokenizer.from_pretrained(esmfold_path)
esmfold_model.esm = esmfold_model.esm.half()
esmfold_model.eval()

# ------------------------------------------------------------
# Helper: Freeze the ProtBERT layers during the first epoch
# ------------------------------------------------------------
def freeze_protbert(model, freeze=True):
    for p in model.protbert.parameters():
        p.requires_grad = not freeze

# ---------------- before training loop ----------------
warmup_head_lr = 1e-5          # head trains fast for 1 epoch
true_critic_lr = args.lr_critic

# freeze both backbones
freeze_protbert(generator, freeze=True)
freeze_protbert(critic,    freeze=True)

# optional: make the head a bit faster for the warm-up epoch
for pg in critic_optimizer.param_groups:
    pg["lr"] = warmup_head_lr

# flag so we un-freeze later
unfreeze_done = False



# ------------------------------------------------------------
# Helper: evaluation + W&B logging
# ------------------------------------------------------------
def run_evaluation(epoch_idx, batch_idx,
                   critic_loss_val, gen_loss_val,
                   tag="eval"):
    """
    Logs structural metrics + current losses to W&B.
    Assumes generator, tokenizer, esmfold_model … are in scope.
    """
    print("=" * 20)
    print(f"[{tag}] Epoch {epoch_idx+1}  Batch {batch_idx} ")

    generated_sequences = generate_fake_sequences(
        generator=generator,
        tokenizer=tokenizer,
        num_sequences=args.num_eval_sequences,
        device=device,
    )

    avg_plddt_score = calculate_plddt_scores_and_save_pdb(
        generated_sequences, esmfold_tokenizer, esmfold_model,
        batch_size=args.eval_batch_size,
        num_sequences=args.num_eval_sequences,
        run_name=args.run_name, device=device
    )
    avg_scAcc = calculate_mpnn_alignment_metric(
        generated_sequences = generated_sequences,
        num_sequences       = args.num_eval_sequences,
        batch_size          = args.eval_batch_size,
        device              = device,
        run_name            = args.run_name
    )

    avg_progres = compute_average_progres_score(
        num_sequences=args.num_eval_sequences, run_name=args.run_name
    )
    avg_pairwise_tm_score = calculate_pairwise_tm_score(
        run_name=args.run_name, num_sequences=args.num_eval_sequences
    )

    wandb.log({
        "epoch":            epoch_idx + 1,
        "batch":            batch_idx,
        "critic_loss":      critic_loss_val,
        "generator_loss":   gen_loss_val,
        "plddt_score":      avg_plddt_score,
        "scAccuracy":       avg_scAcc,
        "progres":          avg_progres,
        "pairwise_tm":      avg_pairwise_tm_score,
        "tag":              tag              # handy for filtering
    })
    clean_m8_folder()


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
        updated_attention_mask = (final_input_ids != tokenizer.mask_token_id).long()
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

    num_crit_batches = len(critic_dataloader)
    mid_batch_idx   = num_crit_batches // 2

    epoch_c_loss_sum = 0.0
    epoch_g_loss_sum = 0.0

    # ----------------- unfreeze ProtBERT after the first epoch -----------------
    if epoch == 1 and not unfreeze_done:
        freeze_protbert(generator, freeze=False)
        freeze_protbert(critic,    freeze=False)

        # drop critic LR to the long-term value
        for pg in critic_optimizer.param_groups:
            pg["lr"] = true_critic_lr

        unfreeze_done = True

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
                batch_size_local=args.batch_size,
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
        epoch_c_loss_sum += critic_loss_val
        epoch_g_loss_sum += g_loss.item()

        if epoch == 0 and batch_number == 0:
            run_evaluation(epoch, 0, critic_loss_val, g_loss.item(), tag="baseline")

        # -------- MID-EPOCH evaluation --------
        if batch_number == mid_batch_idx:
            run_evaluation(epoch, batch_number, critic_loss_val, g_loss.item(), tag="mid")

        batch_number += 1
    
    ### End of epoch logging and saving a checkpoint

    # ================= end-of-epoch =================
    avg_c_epoch = epoch_c_loss_sum / batch_number
    avg_g_epoch = epoch_g_loss_sum / batch_number

    run_evaluation(epoch, batch_number, avg_c_epoch, avg_g_epoch, tag="end")

    # --------------------------
    #  Saving the models
    # --------------------------
    checkpoint_dir = os.path.join(MAIN_FOLDER, "gan-checkpoints")
    save_dir = f"{checkpoint_dir}/{args.run_name}/epoch_{epoch+1}"
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
