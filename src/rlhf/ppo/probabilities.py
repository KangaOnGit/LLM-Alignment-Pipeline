import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

def get_log_prob_sum(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute the summed log probability of a token sequence."""

    outputs = model(input_ids=input_ids)

    logits = outputs.logits[:, :-1]
    labels = input_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)

    token_log_probs = log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    return token_log_probs.sum(dim=-1)