def convert_to_conversational_dpo_format(
    example: dict[str, str]
    ):
    """
    Convert a preference pair into the conversational format expected by DPO.

    Original:
        {
            "question": <user prompt>,
            "chosen": <preferred response>,
            "rejected": <less preferred response>
        }

    Converted:
        {
            "prompt": conversation history,
            "chosen": preferred assistant completion,
            "rejected": non-preferred assistant completion
        }
        
    Maximize:
        max(P(chosen | prompt) > P(rejected | prompt))
            (model learns to prefer chosen over rejected and does not generate answers)

    Directly optimizes the model to assign higher probability to the chosen response
    and lower probability to the rejected response for the same prompt.
    """
    
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