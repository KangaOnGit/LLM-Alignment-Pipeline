import logging

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

log = logging.getLogger(__name__)

def generate_response(
    tokenizer: PreTrainedTokenizer,
    policy: PreTrainedModel,
    prompt: str,
    max_new_tokens: int = 20,
) -> tuple[str, torch.Tensor]:
    """Generate a response from the policy model."""

    log.debug("Generating response for prompt: %s", prompt)

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    with torch.no_grad():
        generated_ids = policy.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    response_ids = generated_ids[:, inputs["input_ids"].shape[-1] :]
    query_response = generated_ids

    response = tokenizer.decode(
        response_ids[0],
        skip_special_tokens=True,
    )

    return response, query_response