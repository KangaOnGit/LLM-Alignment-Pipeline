import re
import logging

from src.utils.config import load_config

log = logging.getLogger(__name__)

CONFIG_GRPO = load_config("configs/rlhf/grpo.yaml")
FORMAT = CONFIG_GRPO["format"]

match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{FORMAT['reasoning_start']}.+?{FORMAT['reasoning_end']}.*?"
    rf"{FORMAT['solution_start']}(.+?){FORMAT['solution_end']}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

# math exactly -> 3.0
def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        if match_format.search(response) is not None:
            score += 3.0
        scores.append(score)
    return scores


def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        score += 0.5 if response.count(FORMAT["reasoning_start"]) == 1 else -1.0
        score += 0.5 if response.count(FORMAT["reasoning_end"]) == 1 else -1.0
        score += 0.5 if response.count(FORMAT["solution_start"]) == 1 else -1.0
        score += 0.5 if response.count(FORMAT["solution_end"]) == 1 else -1.0
        scores.append(score)
    return scores