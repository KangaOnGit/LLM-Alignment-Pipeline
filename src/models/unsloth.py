import torch
from unsloth import FastLanguageModel
import logging

log = logging.getLogger(__name__)

from src.utils.config import load_config, HF_TOKEN

CONFIG_RLHF = load_config("configs/rlhf/grpo.yaml")

DEFAULT_MODEL = CONFIG_RLHF["default"]["model"]

def build_fast_model(
    model_name: str | None = None,
    lora_rank: int = 64,
    max_seq_length: int = 2048,
):
    model_name = model_name or DEFAULT_MODEL

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.8,
    )
    
    return model, tokenizer