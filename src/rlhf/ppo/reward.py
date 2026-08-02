import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)

def compute_reward(
    tokenizer: PreTrainedTokenizer,
    reward_model: PreTrainedModel,
    query_response: torch.Tensor,
) -> torch.Tensor:
    """Compute the reward score for a generated response."""

    reward_inputs = tokenizer(
        tokenizer.decode(
            query_response[0],
            skip_special_tokens=True,
        ),
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    with torch.no_grad():
        reward = reward_model(**reward_inputs).logits.squeeze()

    return reward