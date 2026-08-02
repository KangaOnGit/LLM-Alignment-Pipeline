from transformers import AutoTokenizer
import torch

def formatting_prompt_with_chat_template(
    example: dict[str, str],
    tokenizer: AutoTokenizer,
):
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["chosen"]},
    ]

    return tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False,
    )