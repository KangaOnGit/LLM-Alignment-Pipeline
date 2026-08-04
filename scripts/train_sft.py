import argparse
import logging

import wandb

from datasets import load_dataset
from src.models.huggingface import build_model
from src.sft.trainer import build_sft_trainer
from src.utils.config import HF_TOKEN, load_config, push_hub, get_lora_config

from src.utils.seed import set_seed

set_seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

CONFIG_SFT = load_config("configs/rlhf/sft_dpo.yaml")

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Supervised Fine-Tuning (SFT) model.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=CONFIG_SFT["default"]["model"],
        help="Model name for SFT.",
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
        default="llama-3.2-1b-sft",
        help="Weights & Biases run name.",
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default=CONFIG_SFT["data"]["path"],
        help="Link to Dataset (HuggingFace).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    wandb.init(
        project=args.project,
        name=args.run_name,
        config=CONFIG_SFT,
    )

    model, tokenizer = build_model(args.model)

    log.info("Loading dataset: %s", args.dataset)
    dataset = load_dataset(args.dataset)
    
    log.info("Building LoRA configuration...")
    peft_config = get_lora_config()

    log.info("Building trainer...")
    trainer = build_sft_trainer(
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