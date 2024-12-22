import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

def load_and_tokenize_dataset(tokenizer, gen_file, critic_file, max_length=512):
    """
    Load and tokenize the dataset.
    """
    datasets = load_dataset("text", data_files={
        "gen": gen_file,
        "critic": critic_file
    })

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

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
    gen_dataset = DNMTDataset(tokenized_datasets["gen"])
    gen_dataloader = DataLoader(
        gen_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    critic_dataset = DNMTDataset(tokenized_datasets["critic"])
    critic_dataloader = DataLoader(
        critic_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return gen_dataloader, critic_dataloader

