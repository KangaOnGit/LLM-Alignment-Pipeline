import re

from src.utils.config import load_config
from src.prompts.render import render_prompt


CONFIG_GRPO = load_config("configs/rlhf/grpo.yaml")

REASONING_START = CONFIG_GRPO["format"]["reasoning_start"]
REASONING_END = CONFIG_GRPO["format"]["reasoning_end"]
SOLUTION_START = CONFIG_GRPO["format"]["solution_start"]
SOLUTION_END = CONFIG_GRPO["format"]["solution_end"]


ANSWER_PATTERN = re.compile(
    r"(đáp án là:|đáp án là :|câu trả lời là:|câu trả lời là :)\s*(.*)",
    re.IGNORECASE,
)


SYSTEM_PROMPT = render_prompt(
    "system/default.jinja",
    reasoning_start=REASONING_START,
    reasoning_end=REASONING_END,
    solution_start=SOLUTION_START,
    solution_end=SOLUTION_END,
)


def convert_to_conversational_grpo_format(
    dataset,
    limit: int | None = None,
):
    """
    Convert a mathematical reasoning dataset into the conversational
    format expected by GRPO.

    Original:
        {
            "query_vi": <problem statement>,
            "response_vi": <solution containing final answer>
        }

    Converted:
        {
            "prompt": conversation history,
            "answer": ground-truth answer
        }

    GRPO training process:

        prompt
            |
            v
        policy model generates multiple completions
            |
            v
        reward functions score each generated completion
            |
            v
        GRPO computes relative advantages between generations
            |
            v
        model parameters are updated to increase the probability
        of higher-reward completions

    The answer field is not provided to the model during generation.
    It is only used as a reference by reward functions (e.g., answer
    correctness checking) to compute the reward signal.

        max(R(policy(x), reward functions))
            -> maximize reward of generated responses
            
    prompt + reference answer
    -> generate responses during training and optimize the model
        based on reward feedback from those responses
    """

    formatted_dataset = []

    for item in dataset:

        response = item["response_vi"].strip()

        match = ANSWER_PATTERN.search(response)

        if not match:
            continue

        formatted_dataset.append(
            {
                "prompt": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": item["query_vi"],
                    },
                ],
                "answer": match.group(2).strip(),
            }
        )

        if limit and len(formatted_dataset) >= limit:
            break

    return formatted_dataset