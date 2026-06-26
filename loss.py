import torch


def critic_loss(real_scores, fake_scores, gradient_penalty, lambda_gp=10):
    return (
        -torch.mean(real_scores)
        + torch.mean(fake_scores)
        + lambda_gp * gradient_penalty
    )


def generator_loss(fake_scores):
    return -torch.mean(fake_scores)


def compute_gradient_penalty(
    critic, real_embeds, fake_embeds, real_mask, fake_mask, device
):
    # real_embeds and fake_embeds must be pre-computed [B, L, H] float tensors.
    # Callers handle embedding; interpolation in continuous embedding space is
    # geometrically meaningful, interpolating raw integer token IDs is not.

    alpha = torch.rand(real_embeds.size(0), 1, 1).to(device)

    # Random convex combination between real and fake in embedding space.
    interpolates = (alpha * real_embeds + (1 - alpha) * fake_embeds).requires_grad_(
        True
    )

    # Only attend to positions that are non-padding in BOTH sequences.
    joint_mask = (real_mask.bool() & fake_mask.bool()).float()

    # Run interpolated embeddings through the full transformer (not the critic.forward
    # 3D shortcut, which bypasses the encoder and only hits the classifier head).
    extended_mask = critic.protbert.bert.get_extended_attention_mask(
        joint_mask, interpolates.shape[:2]
    )
    transformer_output = critic.protbert.bert.encoder(
        interpolates, attention_mask=extended_mask
    )
    last_hidden_state = transformer_output.last_hidden_state

    cls_output = last_hidden_state[:, 0, :]
    critic_scores = critic.classifier(cls_output)

    gradients = torch.autograd.grad(
        outputs=critic_scores,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Zero padding positions so they don't inflate the gradient norm.
    joint_mask_expanded = joint_mask.unsqueeze(-1).expand_as(gradients)
    gradients = gradients * joint_mask_expanded

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
