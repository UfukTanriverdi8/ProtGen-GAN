import torch

def critic_loss(real_scores, fake_scores, gradient_penalty, lambda_gp=10):
    return -torch.mean(real_scores) + torch.mean(fake_scores) + lambda_gp * gradient_penalty

def generator_loss(fake_scores):
    return -torch.mean(fake_scores)


def compute_gradient_penalty(critic, real_data, fake_data, real_mask, fake_mask, device):
    # OLD: def compute_gradient_penalty(critic, real_data, fake_data, device)
    # We weren't passing the real/fake attention masks at all.
    # This meant we had no way to know which positions were padding.

    alpha = torch.rand(real_data.size(0), 1, 1).to(device)

    # Get embeddings for real and fake sequences.
    # We have to interpolate at embedding level because real_data and fake_data
    # are token IDs (integers). You can't interpolate integers —
    # averaging token ID 4 and token ID 9 gives you 6.5 which means nothing.
    # Embeddings are continuous vectors so interpolation is meaningful there.
    real_embeds = critic.protbert.bert.embeddings(input_ids=real_data)
    fake_embeds = critic.protbert.bert.embeddings(input_ids=fake_data)

    # Pick a random point between real and fake embeddings.
    # alpha is random per sample in the batch, so we get different in-between
    # points each time. This is how we sample the region between real and fake
    # to check the critic's sensitivity there.
    interpolates = (alpha * real_embeds + (1 - alpha) * fake_embeds).requires_grad_(True)

    # OLD: attention_mask = torch.ones(interpolates.size()[:2]).to(device)
    # This told the transformer "every position is real content" for ALL sequences.
    # But real and fake sequences have different lengths, so their padding is
    # in different positions. The interpolated embeddings in padding regions
    # are meaningless noise (average of a real token and a pad token).
    # Treating that noise as real content distorted the gradient norm calculation.

    # FIX: only keep positions that are real in BOTH sequences.
    # If position 5 is padding in either real or fake, we ignore it.
    # This way the gradient is only measured over meaningful positions.
    joint_mask = (real_mask & fake_mask.bool()).float()

    # OLD: critic_scores = critic(interpolates, attention_mask=attention_mask)
    # This passed the 3D interpolated embeddings directly to critic.forward().
    # The critic's forward method has a dim==3 branch that skips ProtBERT entirely
    # and sends embeddings straight to the classifier head.
    # So the GP was measuring sensitivity of just the classifier MLP —
    # a completely different function from the actual critic which uses
    # 30 layers of transformer attention before the classifier.
    # The Lipschitz constraint was being enforced on the wrong function.

    # FIX: manually run the interpolated embeddings through the full transformer.
    # get_extended_attention_mask converts our 0/1 mask into the format
    # ProtBERT's encoder expects (large negative numbers for ignored positions).
    extended_mask = critic.protbert.bert.get_extended_attention_mask(
        joint_mask, interpolates.shape[:2]
    )

    # Now actually run through all 30 transformer layers.
    # This is the same path the critic takes during real training,
    # so the GP is now measuring the right function.
    transformer_output = critic.protbert.bert.encoder(
        interpolates,
        attention_mask=extended_mask
    )
    last_hidden_state = transformer_output.last_hidden_state

    # Take CLS token and run through classifier — same as normal critic forward pass.
    cls_output = last_hidden_state[:, 0, :]
    critic_scores = critic.classifier(cls_output)

    # Compute how much critic_scores change when we nudge the interpolated embeddings.
    # This gives us the gradient of the critic at this interpolated point.
    gradients = torch.autograd.grad(
        outputs=critic_scores,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Zero out gradients at padding positions before computing the norm.
    # Even though we masked them in the transformer, the gradient tensor
    # still has values at those positions. We zero them out so they don't
    # contribute to the norm calculation.
    joint_mask_expanded = joint_mask.unsqueeze(-1).expand_as(gradients)
    gradients = gradients * joint_mask_expanded

    # Flatten and compute the norm per sample in the batch.
    # Then penalize how far the norm is from 1.
    # This is the actual penalty term — pushing the critic toward norm=1
    # which keeps it calibrated (not too confident, not too unsure).
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty