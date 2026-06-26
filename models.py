import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(
        self,
        protbert_model,
        cls_token_id=2,
        sep_token_id=3,
        mask_token_id=4,
        pad_token_id=0,
    ):
        super().__init__()
        self.protbert = protbert_model
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id

    def forward(self, input_ids, attention_mask=None):
        outputs = self.protbert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=1.0,
        keep_percent=0.1,
        current_rate=None,
    ):
        batch_size, seq_len = input_ids.size()
        outputs = self.protbert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits / temperature
        probabilities = F.softmax(logits, dim=-1)

        predicted_ids = torch.multinomial(
            probabilities.view(-1, probabilities.size(-1)), num_samples=1
        ).view(batch_size, seq_len)
        confidence = probabilities.gather(-1, predicted_ids.unsqueeze(-1)).squeeze(-1)

        # ☣️ NUCLEAR MISTAKE - this caused hours of trainings to be wasted because
        # it was just picking the most confident token for each position, which is not what we want at all.
        # We want to sample from the distribution and then use those sampled tokens to determine which masked positions to fill.
        # confidence, predicted_ids = probabilities.max(dim=-1)
        # Let us pay our respects to the fallen gpu hours for a moment of silence 🪦

        for i in range(batch_size):
            seq_mask_indices = input_ids[i] == self.mask_token_id
            if not seq_mask_indices.any():
                continue

            # If we're on (or past) the final iteration, fill everything
            remaining_masks = seq_mask_indices.sum().item()
            if current_rate is not None and current_rate <= 0.1:
                # Fill all remaining
                num_to_fill = remaining_masks
            else:
                meaningful_seq = (
                    (input_ids[i] != self.pad_token_id)
                    & (input_ids[i] != self.cls_token_id)
                    & (input_ids[i] != self.sep_token_id)
                )
                meaningful_count = meaningful_seq.sum().item()
                num_to_fill = max(1, int(keep_percent * meaningful_count))
                remaining_masks = seq_mask_indices.sum().item()
                num_to_fill = min(num_to_fill, remaining_masks)

            # Now pick the top-k for only this sequence
            seq_confidence = confidence[i].clone()
            seq_confidence[~seq_mask_indices] = 0.0

            topk_values, topk_positions = torch.topk(seq_confidence, num_to_fill)
            for pos in topk_positions.cpu().tolist():  # ‹- convert to list of ints
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
            nn.Linear(hidden_size // 16, 1),
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
            extended_mask = self.protbert.bert.get_extended_attention_mask(
                attention_mask, input_data.shape[:2]
            )
            transformer_output = self.protbert.bert.encoder(
                input_data, attention_mask=extended_mask
            )
            last_hidden_state = transformer_output.last_hidden_state

        cls_output = last_hidden_state[:, 0, :]  # CLS token embedding
        logits = self.classifier(cls_output)
        return logits


def compute_soft_embeds(generator, critic, input_ids, attn_mask, min_temp, max_temp):
    """One generator forward pass → continuous embeddings for the critic (K=1 backprop path).

    Gradient flows: critic(soft_embeds) → loss → backward → probs → logits → generator.protbert.
    The iterative fill loop that produced input_ids is NOT in this graph — only this call is.

    Returns (soft_embeds, probs, temperature). probs and temperature are needed by the KL anchor
    so the reference model can be evaluated at the same temperature without a second gen forward pass.
    """
    temperature = min_temp + torch.rand(1).item() * (max_temp - min_temp)
    logits = generator(input_ids, attn_mask)  # [B, L, V]
    probs = F.softmax(logits / temperature, dim=-1)  # [B, L, V]
    word_weight = critic.protbert.bert.embeddings.word_embeddings.weight  # [V, H]
    soft_word = probs @ word_weight  # [B, L, H]
    soft_embeds = critic.protbert.bert.embeddings(inputs_embeds=soft_word)  # [B, L, H]
    return soft_embeds, probs, temperature


def compute_kl_anchor(gen_probs, ref_protbert, input_ids, attn_mask, temperature):
    """KL(generator || frozen_reference) — prevents the generator from reward-hacking.

    Uses the same temperature as compute_soft_embeds so both distributions are comparable.
    ref_protbert must be frozen (requires_grad=False, eval mode) — never updated.
    """
    with torch.no_grad():
        ref_logits = ref_protbert(input_ids=input_ids, attention_mask=attn_mask).logits
        ref_log_probs = F.log_softmax(ref_logits / temperature, dim=-1)  # [B, L, V]
    # Clamp before kl_div: F.kl_div computes log(gen_probs) internally; without the
    # clamp, near-zero probs → log(~0) ≈ -87 → giant gradients → weight explosion.
    return F.kl_div(
        ref_log_probs,
        gen_probs.clamp(min=1e-8),
        reduction="batchmean",
        log_target=False,
    )
