from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import GRPOConfig, GRPOTrainer
from typing import Callable

from src.utils.config import load_config

CONFIG = load_config("configs/rlhf/grpo.yaml")
TRAIN_CFG = CONFIG["train"]
OUTPUT_CFG = CONFIG["output"]


def get_grpo_config(**overrides) -> GRPOConfig:
    config = {
        "output_dir": OUTPUT_CFG["path"],
        "learning_rate": TRAIN_CFG["lr"],
        "weight_decay": TRAIN_CFG["weight_decay"],
        "warmup_ratio": TRAIN_CFG["warmup_ratio"],
        "lr_scheduler_type": TRAIN_CFG["lr_scheduler"],
        "optim": TRAIN_CFG["optim"],
        "logging_steps": TRAIN_CFG["logging_steps"],
        "per_device_train_batch_size": TRAIN_CFG["batch_size"],
        "gradient_accumulation_steps": TRAIN_CFG["gradient_steps"],
        "num_generations": TRAIN_CFG["num_generations"],
        "max_prompt_length": TRAIN_CFG["max_prompt_length"],
        "max_completion_length": TRAIN_CFG["max_completion_length"],
        "num_train_epochs": TRAIN_CFG["epochs"],
        "max_steps": TRAIN_CFG["max_steps"],
        "save_steps": TRAIN_CFG["save_steps"],
        "max_grad_norm": TRAIN_CFG["max_grad_norm"],
        "report_to": TRAIN_CFG["report_to"],
    }

    config.update(overrides)
    return GRPOConfig(**config)

def build_grpo_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    reward_funcs: list[Callable],
    **config_overrides,
) -> GRPOTrainer:
    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        train_dataset=dataset,
        args=get_grpo_config(**config_overrides),
    )