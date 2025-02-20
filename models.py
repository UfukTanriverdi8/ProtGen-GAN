import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation.utils import GenerationMixin

class Generator(nn.Module):
    def __init__(self, protbert_model, cls_token_id=2, sep_token_id=3 , mask_token_id=4, pad_token_id=0):
        super().__init__()
        self.protbert = protbert_model
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id

    def forward(self, input_ids, attention_mask=None):
        outputs = self.protbert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    """def generate(self, input_ids, attention_mask=None, temperature=1.0, keep_percent=0.1, current_rate=None):
        meaningful_seq = (input_ids != self.pad_token_id) & (input_ids != self.cls_token_id) & (input_ids != self.sep_token_id)

        meaningful_token_count = meaningful_seq.sum().item()  # the keep_percent will be calculated by this value

        outputs = self.protbert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits / temperature

        probabilities = F.softmax(logits, dim=-1)
        confidence, predicted_ids = probabilities.max(dim=-1)

        mask_indices = (input_ids == self.mask_token_id)
        confidence[~mask_indices] = 0.0  # we will only get the confidence for the mask indices

        num_tokens_to_fill = max(1, int(keep_percent * meaningful_token_count))  # Constant 10% fill
        remaining_masks = mask_indices.sum().item()

        if remaining_masks == 0:
            print("All tokens are filled early!")
            return input_ids  # No more masks to fill
        
        if current_rate == 0.1:
            num_tokens_to_fill = remaining_masks
        else:
            num_tokens_to_fill = min(num_tokens_to_fill, remaining_masks)  # don't fill more than available

        # Get top-k confident predictions
        topk_indices = torch.topk(confidence.view(-1), num_tokens_to_fill).indices

        batch_indices = topk_indices // input_ids.size(1)
        seq_indices = topk_indices % input_ids.size(1)

        for batch_idx, seq_idx in zip(batch_indices, seq_indices):
            if input_ids[batch_idx, seq_idx] == self.mask_token_id:
                input_ids[batch_idx, seq_idx] = predicted_ids[batch_idx, seq_idx]

        return input_ids """
    def generate(self, input_ids, attention_mask=None, temperature=1.0, keep_percent=0.1, current_rate=None):
        batch_size, seq_len = input_ids.size()
        outputs = self.protbert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits / temperature
        probabilities = F.softmax(logits, dim=-1)
        confidence, predicted_ids = probabilities.max(dim=-1)

        for i in range(batch_size):
            seq_mask_indices = (input_ids[i] == self.mask_token_id)
            if not seq_mask_indices.any():
                continue

            # If we're on (or past) the final iteration, fill everything
            remaining_masks = seq_mask_indices.sum().item()
            if current_rate is not None and current_rate <= 0.1:
                # Fill all remaining
                num_to_fill = remaining_masks
            else:
                # Normal top-k fill
                # Compute how many tokens to fill (10% of meaningful tokens, etc.)
                # ...
                meaningful_seq = (
                    (input_ids[i] != self.pad_token_id) &
                    (input_ids[i] != self.cls_token_id) &
                    (input_ids[i] != self.sep_token_id)
                )
                meaningful_count = meaningful_seq.sum().item()
                num_to_fill = max(1, int(keep_percent * meaningful_count))
                remaining_masks = seq_mask_indices.sum().item()
                num_to_fill = min(num_to_fill, remaining_masks)
                num_to_fill = min(num_to_fill, remaining_masks)

            # Now pick the top-k for only this sequence
            seq_confidence = confidence[i].clone()
            seq_confidence[~seq_mask_indices] = 0.0

            topk_values, topk_positions = torch.topk(seq_confidence, num_to_fill)
            for pos in topk_positions:
                if input_ids[i, pos] == self.mask_token_id:
                    input_ids[i, pos] = predicted_ids[i, pos]

        return input_ids




class Critic(nn.Module):
    def __init__(self, protbert_model):
        super().__init__()
        self.protbert = protbert_model
        hidden_size = self.protbert.config.hidden_size

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
