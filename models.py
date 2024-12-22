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
        hidden_size = self.protbert.config.hidden_size  # e.g., 1024

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
