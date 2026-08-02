import torch

def ppo_loss(
    logprob_policy: torch.Tensor,
    logprob_reference: torch.Tensor,
    advantage: torch.Tensor,
    clip_eps: float = 0.2,
    kl_coef: float = 0.01,
) -> torch.Tensor:
    """Compute the PPO clipped loss with a KL penalty."""

    log_ratio = logprob_policy - logprob_reference
    ratio = torch.exp(log_ratio)

    unclipped_objective = ratio * advantage
    clipped_objective = (
        torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        * advantage
    )

    ppo_loss = -torch.min(
        unclipped_objective,
        clipped_objective,
    ).mean()

    kl_loss = log_ratio.square().mean()

    return ppo_loss + kl_coef * kl_loss