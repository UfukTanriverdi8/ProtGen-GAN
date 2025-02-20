import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

def load_and_tokenize_dataset(
    tokenizer,
    gen_file=None,
    critic_file=None,
    full_dataset=None,
    fully_masked=False,
    max_length=512
):
    if fully_masked:
        # Load a single "full" dataset file and treat it as the critic dataset.
        datasets = load_dataset("text", data_files={"critic": full_dataset})
    else:
        datasets = load_dataset("text", data_files={"gen": gen_file, "critic": critic_file})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_datasets

class DNMTDataset(Dataset):
    def __init__(self, tokenized_data, mask_token_id=4):
        self.input_ids = tokenized_data["input_ids"]
        self.attention_mask = tokenized_data["attention_mask"]
        self.mask_token_id = mask_token_id

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        attention_mask = torch.tensor(self.attention_mask[idx], dtype=torch.float)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

def get_dataloaders(tokenized_datasets, batch_size):
    # If fully_masked mode is active, the tokenized_datasets will only have "critic"
    if "gen" in tokenized_datasets:
        gen_dataset = DNMTDataset(tokenized_datasets["gen"])
        gen_dataloader = DataLoader(
            gen_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
    else:
        # In fully masked mode, use the critic dataset for both generator and critic.
        gen_dataloader = None

    critic_dataset = DNMTDataset(tokenized_datasets["critic"])
    critic_dataloader = DataLoader(
        critic_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    if gen_dataloader is None:
        return critic_dataloader

    return gen_dataloader, critic_dataloader
