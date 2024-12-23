import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, protbert_model, mask_token_id = 4):
        super().__init__()
        self.protbert = protbert_model
        self.mask_token_id = mask_token_id

    def forward(self, input_ids, attention_mask=None):
        outputs = self.protbert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
    def generate(self, input_ids, attention_mask=None, temperature=1.0, keep_percent=0.1):
        # not including special tokens except masks
        meaningful_mask = (input_ids >= self.mask_token_id)

        # logits
        outputs = self.protbert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # temperature scaling and softmax
        logits = logits / temperature
        probabilities = F.softmax(logits, dim=-1)

        # compute confidence and predicted IDs
        confidence, predicted_ids = probabilities.max(dim=-1)

        # Zero out confidence for non-meaningful tokens(cls, unk, sep, pad)
        confidence[~meaningful_mask] = 0.0

        # top percent that will be kept
        total_meaningful_tokens = meaningful_mask.sum().item()
        num_tokens_to_keep = int(keep_percent * total_meaningful_tokens)

        # Get top-k confident token indices
        topk_indices = torch.topk(confidence.view(-1), num_tokens_to_keep).indices

        # Convert flat indices back to 2D indices (batch, seq)
        batch_indices = topk_indices // input_ids.size(1)
        seq_indices = topk_indices % input_ids.size(1)

        # Replace only `[MASK]` tokens in the top 10%
        for batch_idx, seq_idx in zip(batch_indices, seq_indices):
            if input_ids[batch_idx, seq_idx] == self.mask_token_id:  # Only replace `[MASK]` tokens
                input_ids[batch_idx, seq_idx] = predicted_ids[batch_idx, seq_idx]

        return input_ids



class Critic(nn.Module):
    def __init__(self, protbert_model):
        super().__init__()
        self.protbert = protbert_model
        hidden_size = self.protbert.config.hidden_size  # 1024

        # Classification head
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
            last_hidden_state = self.protbert.bert.embeddings(input_data)  # Convert to embeddings
        elif input_data.dim() == 3:  # Already Embeddings
            last_hidden_state = input_data

        cls_output = last_hidden_state[:, 0, :]  # CLS token embedding
        logits = self.classifier(cls_output)
        return logits
