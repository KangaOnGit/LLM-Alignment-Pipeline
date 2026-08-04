import re
import logging

from src.utils.config import load_config
from src.rlhf.grpo.reward.match_format import match_format

log = logging.getLogger(__name__)

CONFIG_GRPO = load_config("configs/rlhf/grpo.yaml")
FORMAT = CONFIG_GRPO["format"]

match_numbers = re.compile(
    FORMAT["solution_start"] + r".*?(-?[\d\.\,]{1,})",
    flags=re.MULTILINE | re.DOTALL
)

def check_answer(prompts, completions, answer, **kwargs):
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        
        if guess == true_answer:
            score += 3.0
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            score -= 1.5
        scores.append(score)
    return scores

def check_numbers(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_numbers.search(r)) is not None else None
        for r in responses
    ]

    # Print every 5 steps
    count = getattr(check_numbers, 'counter', 0) + 1
    check_numbers.counter = count
    if count % 5 == 0:
        log.info('*'*20, f"Question:{question}", f"\nResponse:\n{responses[0]}",
              f"\nExtracted: {extracted_responses[0]}", f"\nGT Answer: {answer[0]}")

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0)
            continue
        # Convert to numbers
        try:
            true_answer = float(true_answer.strip())
            guess = float(guess.strip().replace(",", ""))
            scores.append(1.5 if guess == true_answer else -0.5)
        except:
            scores.append(0)
    return scores