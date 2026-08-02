from pathlib import Path
import yaml
import os
import bitsandbytes as bnb
import torch

from dotenv import load_dotenv
from pathlib import Path
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer
from typing import Sequence



load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_config(path):
    with open(PROJECT_ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_bnb_config(
    load_4bit: bool = True,
    quant_type: str = "nf4",
    double_quant: bool = True,
):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_4bit,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=double_quant,
        )
    return bnb_config

def get_qlora_config(
    r: int=16,
    lora_alpha: int=16,
    lora_dropout: float=0.05,
    bias: str="none",
    task_type: str="CAUSAL_LM",
    target_modules: Sequence[str] | None = None,
):
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
        target_modules=target_modules,)

def push_hub(
    name: str,
    trainer: SFTTrainer,
) -> None:
    trainer.push_to_hub(name, token=HF_TOKEN)
    
if __name__ == "__main__":
    from huggingface_hub import scan_cache_dir
    print(scan_cache_dir())