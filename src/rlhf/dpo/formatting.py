def convert_to_conversational_preference_format(
    example: dict[str, str]
    ):
    return {
        "id": example["id"],
        "prompt": [{"role": "system",
                    "content": "You are a helpful assistant."},
                   {"role": "user",
                    "content": example["question"]}],
        "chosen": [{"role": "assistant",
                    "content": example["chosen"]}],
        "rejected": [{"role": "assistant",
                      "content": example["rejected"]}],
    }