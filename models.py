import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, protbert_model):
        super().__init__()
        self.protbert = protbert_model

    def forward(self, input_ids, attention_mask=None):
        outputs = self.protbert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
    def generate(self, input_ids, attention_mask=None, temperature=1.0, keep_percent=0.1):
        # meaningful tokens
        meaningful_mask = (input_ids > 4)

        for _ in range(input_ids.size(1)):  # Iterate up to sequence length
            outputs = self.protbert(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Apply temperature scaling
            logits = logits / temperature
            probabilities = F.softmax(logits, dim=-1)

            # Compute confidence (max probability for each token)
            confidence, predicted_ids = probabilities.max(dim=-1)

            # Set confidence to zero for unmeaningful tokens
            confidence = confidence * meaningful_mask.float()

            # Process each sequence in the batch
            retain_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for batch_idx in range(meaningful_mask.size(0)):
                num_tokens_to_keep = int(keep_percent * meaningful_mask[batch_idx].sum().item())
                topk_indices = torch.topk(confidence[batch_idx], num_tokens_to_keep).indices

                # Create retain_mask for the current sequence
                retain_mask[batch_idx].scatter_(0, topk_indices, True)

            # Update input_ids for the entire batch
            fill_mask = ~retain_mask & (input_ids <= 4)
            input_ids[fill_mask] = predicted_ids[fill_mask]

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
