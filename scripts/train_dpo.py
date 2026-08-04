import argparse
import logging

import wandb

from datasets import load_dataset
from src.models.huggingface import build_model
from src.rlhf.dpo.formatting import convert_to_conversational_preference_format
from src.rlhf.dpo.trainer import build_dpo_trainer
from src.utils.config import HF_TOKEN, load_config, push_hub, get_lora_config

from src.utils.seed import set_seed

set_seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

CONFIG_DPO = load_config("configs/rlhf/sft_dpo.yaml")

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DPO model.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=CONFIG_DPO["default"]["model"],
        help="Model name for DPO.",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="vi-alpaca-preference",
        help="Weights & Biases project name.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="llama-3.2-1b-dpo",
        help="Weights & Biases run name.",
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default=CONFIG_DPO["data"]["path"],
        help="Link to Dataset (HuggingFace)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    wandb.init(
        project=args.project,
        name=args.run_name,
        config=CONFIG_DPO,
    )

    model, tokenizer = build_model(args.model)

    log.info("Loading dataset: %s", args.dataset)
    dataset = load_dataset(args.dataset)
    dpo_dataset = dataset.map(convert_to_conversational_preference_format)

    log.info("Building LoRA configuration...")
    peft_config = get_lora_config()

    log.info("Building trainer...")
    trainer = build_dpo_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        peft_config=peft_config,
    )

    log.info("Starting training...")
    try:
        trainer.train()

        log.info("Training complete.")

        push_hub(
            name=args.run_name,
            trainer=trainer,
            token=HF_TOKEN,
        )
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()