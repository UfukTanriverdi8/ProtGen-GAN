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
        # Create a mask for meaningful tokens (not 0, 1, 2, 3, 4)
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

            # Determine top 10% confident tokens to retain
            num_tokens_to_keep = int(keep_percent * meaningful_mask.sum(dim=1).item())
            topk_indices = torch.topk(confidence, num_tokens_to_keep, dim=-1).indices

            # Create a mask to retain only the top 10% most confident tokens
            retain_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            retain_mask.scatter_(1, topk_indices, True)

            # Update input_ids: replace only unmasked, non-retained tokens
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
