import json
import re
from collections import Counter

import torch
from transformers import pipeline

from src.utils.seed import set_seed


def load_math_dataset(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def build_inference_pipeline(model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
    set_seed(42)
    return pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


def normal_prompt(question: str, system_prompt: str = "You are a helpful math assistant."):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Problem: {question}"},
    ]


def chain_of_thought_prompt(question: str):
    system_prompt = (
        "You will be given a math problem. Think step by step to solve it and "
        "show your intermediate reasoning. Answer in the form: `Answer: <single number or expression>`."
    )
    return normal_prompt(question, system_prompt)


def tree_of_thought_prompt(question: str):
    prompt = (
        "Imagine three different experts are independently solving this question. "
        "All experts will write down 1 step of their thinking, then share it with the group. "
        "Then all experts will go on to the next step, etc. If any expert realises they're wrong "
        "at any point, then they leave."
    )
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"The question is: {question}"},
    ]


def extract_answer(text: str) -> str:
    if "Answer:" in text:
        answer = text.split("Answer:")[-1].strip().split()[0]
    else:
        answer = text.split()[-1]
        answer = re.sub(r"[^0-9]+", "", answer)
    return answer


def self_consistent_answer(pipe, question: str, samples: int = 10) -> str:
    answers = []
    for _ in range(samples):
        messages = chain_of_thought_prompt(question)
        output = pipe(messages, max_new_tokens=2000, do_sample=True)
        text = output[0]["generated_text"][-1]["content"].strip()
        answers.append(extract_answer(text))
    return Counter(answers).most_common(1)[0][0]
