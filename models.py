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


def compute_soft_embeds(
    generator,
    critic,
    input_ids,
    attn_mask,
    temperature,
    tokenizer,
    remask_frac=0.5,
):
    # -- Phase 1: Saving the positions that will be REMASKED later
    # We are saving them from now on so that we can use them before the critic and also for the KL anchor computation
    # TODO: remask_frac is fixed 0.5 currently, but we can explore the idea of using the same prob distribution that we used during finetuning
    remask_positions = torch.rand_like(input_ids, dtype=torch.float) < remask_frac
    # don't mask padding or special tokens (CLS/SEP/PAD)
    special = (
        (input_ids == tokenizer.pad_token_id)
        | (input_ids == tokenizer.cls_token_id)
        | (input_ids == tokenizer.sep_token_id)
    )
    remask_positions = (
        remask_positions & ~special
    )  # [Batch, Length] bool: True = blanked

    # -- Phase 2: Creating the masked input for the generator and running it through the generator

    # masking happens here, we apply it to the copy of the input_ids
    masked_input = input_ids.clone()
    masked_input[remask_positions] = tokenizer.mask_token_id

    # Throwing the masked input through the generator to get the logits and probabilities for the masked positions
    logits = generator(
        masked_input, attn_mask
    )  # [Batch, Length, VocabSize], real choices at masked positions, garbage elsewhere
    probs = F.softmax(logits / temperature, dim=-1)
    word_weight = critic.protbert.bert.embeddings.word_embeddings.weight
    soft_sequence = (
        probs @ word_weight
    )  # [Batch, Length, HiddenSize], every aminoacid is now a weighted sum of the embeddings
    # since every aminoacid is soft now, we have to cherry-pick the actual choices for the masked positions
    # and use the hard embeddings for the unmasked positions

    # -- Phase 3: Blending the soft amino acid embeddings with the original embeddings for unmasked positions

    # Settled sequence stands for the original embeddings of the sequence, without no softening
    settled_sequence = critic.protbert.bert.embeddings.word_embeddings(
        input_ids
    )  # [Batch, Length, HiddenSize]
    # We had the remask_positions as [Batch, Length] bool, we need to unsqueeze it to [Batch, Length, 1] to broadcast over the HiddenSize dimension
    remasked_aminoacids = remask_positions.unsqueeze(
        -1
    )  # [Batch, Length, 1] to broadcast over HiddenSize
    # Blend happens here. Wherever remasked_aminoacids is True, we take the soft_sequence, otherwise we take the settled_sequence
    # Means that for the positions that were chosen to be remasked, we take the soft amino acid embeddings,
    # and for the positions that were not chosen to be remasked, we take the original embeddings of the amino acids
    blended_sequence = torch.where(remasked_aminoacids, soft_sequence, settled_sequence)
    # Adding the positional information to the blended_sequence, and finalizing the sequence that can be read by the critic
    soft_embeds = critic.protbert.bert.embeddings(inputs_embeds=blended_sequence)
    # returning the soft embeddings, probs of generator, temperature used, remask positions and the masked input for KL anchor computation
    return soft_embeds, probs, temperature, remask_positions, masked_input


def compute_confidence_metrics(gen_probs, remask_positions):
    """Avg max-probability and avg entropy of gen_probs, restricted to remasked positions.

    Diagnostic for the confidence-inflation exploit described in
    docs/GENERATOR_GRADIENT_FIX.md (Open risks / caveats): the generator can raise its
    critic score by sharpening its output distribution rather than by becoming more
    DNMT-like, since a peakier distribution pulls compute_soft_embeds' soft blend closer
    to a hard embedding regardless of which token it's confident about. Climbing
    max-probability / falling entropy without a matching rise in quality metrics
    (pLDDT, scAccuracy, unique_ratio) is the signature to watch for.
    """
    remasked_probs = gen_probs[remask_positions]  # [N_masked, VocabSize]
    if remasked_probs.numel() == 0:
        zero = torch.zeros((), device=gen_probs.device, dtype=gen_probs.dtype)
        return zero, zero
    max_prob = remasked_probs.max(dim=-1).values.mean()
    entropy = -(remasked_probs * remasked_probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
    return max_prob, entropy


def compute_kl_anchor(
    gen_probs, ref_protbert, masked_input, attn_mask, temperature, remask_positions
):
    """KL(generator || reference_finetuned_protbert), computed ONLY at remasked positions.

    Once the generator finally gets gradient, it will chase whatever
    fools the critic, including garbage sequences. This anchors it to the frozen
    pretrained ProtBERT so it can't drift into nonsense (the collapse we see from the DRAKES paper(ICLR, 2025)).

    F.kl_div(input, target) computes KL(target || input).
    We pass input = ref_log_probs, target = gen_probs  ->  KL(gen || ref). Forward KL.
    "target is the distribution being measured, input is the reference."

    masked_input: the SAME masked input compute_soft_embeds fed the generator. The
      reference MUST see the identical input, otherwise the two distributions are
      answering different questions and the KL is meaningless.
    remask_positions: [Batch, Length] bool, True where a real prediction happened.
      Only these slots enter the KL. Kept slots are echo logits and would pollute it.
    """
    # -- Phase 1: run the frozen reference finetuned protbert on the SAME masked input
    with torch.no_grad():
        ref_logits = ref_protbert(
            input_ids=masked_input, attention_mask=attn_mask
        ).logits.float()
        # same temperature as the generator's softmax, so the two are comparable
        # using log_softmax here because F.kl_div expects log-probs for the input!
        ref_log_probs = F.log_softmax(
            ref_logits / temperature, dim=-1
        )  # [Batch, Length, VocabSize]

    # -- Phase 2: keep only the remasked slots, flatten [Batch, Length, VocabSize] -> [N_masked, VocabSize]
    # remask_positions picks out the rows where either model actually predicted something
    generator_selected_probs = gen_probs[remask_positions]  # [N_masked, VocabSize]
    ref_selected_probs = ref_log_probs[remask_positions]  # [N_masked, VocabSize]

    # if a batch happened to mask zero real positions, there is nothing to anchor.
    # return a real zero (not NaN) so g_loss stays clean and the step isn't silently skipped.
    if generator_selected_probs.numel() == 0:
        return torch.zeros((), device=gen_probs.device, dtype=gen_probs.dtype)

    # -- Phase 3: forward KL over the masked positions
    # clamping is needed because F.kl_div internally does target * log(target); near-zero probs would
    # make log(~0) blow up. clamp floors the prob so the log stays finite.
    # batchmean here divides by N_masked (dim 0 is now masked-position count), giving
    # per-masked-position KL, whcih is stable no matter how many positions got masked.
    return F.kl_div(
        ref_selected_probs,  # input: log-probs of the reference
        generator_selected_probs.clamp(min=1e-8),  # target: probs of the generator
        reduction="batchmean",
        log_target=False,
    )
