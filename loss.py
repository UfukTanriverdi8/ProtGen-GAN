import torch

def critic_loss(real_scores, fake_scores, gradient_penalty, lambda_gp=10):
    return -torch.mean(real_scores) + torch.mean(fake_scores) + lambda_gp * gradient_penalty

def generator_loss(fake_scores):
    return -torch.mean(fake_scores)


def compute_gradient_penalty(critic, real_data, fake_data, device):
    alpha = torch.rand(real_data.size(0), 1, 1).to(device) 
    real_embeds = critic.protbert.bert.embeddings(real_data)  
    fake_embeds = critic.protbert.bert.embeddings(fake_data)  
    interpolates = (alpha * real_embeds + (1 - alpha) * fake_embeds).requires_grad_(True)  
    attention_mask = torch.ones(interpolates.size()[:2]).to(device)  # Full mask (no padding)

    # Pass interpolated embeddings through the critic
    critic_scores = critic(
        interpolates,
        attention_mask=attention_mask
    )

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=critic_scores,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)  # Flatten
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()  # Gradient penalty
    return gradient_penalty