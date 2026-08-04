from transformers import BitsAndBytesConfig
import torch


def get_bnb_config(
    load_4bit: bool = True,
    quant_type: str = "nf4",
    double_quant: bool = True,
) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=load_4bit,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=double_quant,
    )