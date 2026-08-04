from datasets import DatasetDict
from peft import LoraConfig
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import DPOConfig, DPOTrainer

from src.rlhf.dpo.formatting import formatting_prompt_with_chat_template
from src.utils.config import load_config

CONFIG = load_config("configs/rlhf/dpo.yaml")

TRAIN_CFG = CONFIG["train"]
DPO_CFG = CONFIG["dpo"]
OUTPUT_CFG = CONFIG["output"]


def get_dpo_config(**overrides) -> DPOConfig:
    config = {
        "output_dir": OUTPUT_CFG["dir"],

        # Training
        "per_device_train_batch_size": TRAIN_CFG["batch_size"],
        "gradient_accumulation_steps": TRAIN_CFG["gradient_steps"],
        "gradient_checkpointing": TRAIN_CFG["gradient_checkpointing"],

        # Optimization
        "learning_rate": TRAIN_CFG["lr"],
        "optim": TRAIN_CFG["optim"],
        "warmup_steps": TRAIN_CFG["warmup_steps"],

        # Schedule
        "num_train_epochs": TRAIN_CFG["epochs"],

        # Logging / Saving
        "logging_steps": TRAIN_CFG["logging_steps"],
        "save_strategy": TRAIN_CFG["save_strategy"],
        "overwrite_output_dir": TRAIN_CFG["overwrite_output_dir"],

        # Precision
        "bf16": TRAIN_CFG["bf16"],

        # Sequence
        "max_length": TRAIN_CFG["max_length"],

        # DPO
        "beta": DPO_CFG["beta"],
    }

    config.update(overrides)

    return DPOConfig(**config)


def build_dpo_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: DatasetDict,
    peft_config: LoraConfig,
    **config_overrides,
) -> DPOTrainer:
    return DPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        args=get_dpo_config(**config_overrides),
        formatting_func=formatting_prompt_with_chat_template,
    )