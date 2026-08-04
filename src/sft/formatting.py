def convert_to_conversational_sft_format(
    example: dict[str, str],
):
    """
    Convert a supervised fine-tuning example into the conversational
        message format expected by the model

    Original:
        {
            "question": <user prompt>,
            "chosen": <target assistant response>
        }

    Converted:
        {
            "messages": [
                system -> defines assistant behavior,
                user -> input prompt,
                assistant -> ground-truth response
            ]
        }

    Model learns to maximize the likelihood of the
    assistant response given the system and user messages:
    
        max(P(assistant_response/chosen response | system_prompt, user_prompt))
            (model learns to imitate the answer, with the answer already exists)
        
    Unlike DPO, there is no preference comparison. The "chosen" response
    is treated as the correct target output.
    """

    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": example["question"],
            },
            {
                "role": "assistant",
                "content": example["chosen"],
            },
        ]
    }