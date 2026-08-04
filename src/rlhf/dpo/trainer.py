from datasets import DatasetDict
from peft import LoraConfig
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import DPOConfig, DPOTrainer

from src.rlhf.dpo.formatting import formatting_prompt_with_chat_template
from src.utils.config import load_config

CONFIG = load_config("configs/rlhf/sft_dpo.yaml")
TRAIN_CFG = CONFIG["train"]
OUTPUT_CFG = CONFIG["output"]


def get_dpo_config(**overrides) -> DPOConfig:
    config = {
        "output_dir": OUTPUT_CFG["path"],
        "per_device_train_batch_size": TRAIN_CFG["batch_size"],
        "gradient_accumulation_steps": TRAIN_CFG["steps"],
        "learning_rate": TRAIN_CFG["lr"],
        "logging_steps": TRAIN_CFG["log_steps"],
        "num_train_epochs": TRAIN_CFG["epochs"],
        "save_strategy": TRAIN_CFG["save"],
        "overwrite_output_dir": TRAIN_CFG["overwrite_output_dir"],
        "optim": TRAIN_CFG["optim"],
        "warmup_steps": TRAIN_CFG["warmup_steps"],
        "bf16": TRAIN_CFG["bf16"],
        "max_length": TRAIN_CFG["max_length"],
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