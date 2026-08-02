import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import logging

log = logging.getLogger(__name__)

from src.utils.config import get_bnb_config, load_config, HF_TOKEN

CONFIG_SFT = load_config("configs/sft.yaml")

DEFAULT_MODEL = CONFIG_SFT["default"]["model"]
CACHE_DIR = CONFIG_SFT["cache"]["path"]


def build_model(
    model_name: str | None = None,
) -> PreTrainedModel:
    model_name = model_name or DEFAULT_MODEL

    log.info("Loading model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map={"": torch.cuda.current_device()},
        token=HF_TOKEN,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        quantization_config=get_bnb_config(),
    )
    log.info("Loading %s successful.", model_name)

    model.config.use_cache = False
    return model


def build_tokenizer(
    model_name: str | None = None,
) -> PreTrainedTokenizer:
    model_name = model_name or DEFAULT_MODEL

    log.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        token=HF_TOKEN
    )
    log.info("Loading %s successful.", model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer