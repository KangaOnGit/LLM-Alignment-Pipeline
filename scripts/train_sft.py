import argparse
import logging

import wandb
from datasets import load_dataset

from src.models.huggingface import build_model
from src.peft.config import get_lora_config
from src.sft.trainer import build_sft_trainer

from src.utils.config import load_config
from src.utils.hub import push_hub
from src.utils.seed import set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

log = logging.getLogger(__name__)

CONFIG_SFT = load_config("configs/rlhf/sft.yaml")


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
        default=CONFIG_SFT["output"]["hub_name"],
        help="Weights & Biases run name / Hugging Face Hub repository name.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=CONFIG_SFT["data"]["path"],
        help="Hugging Face dataset name or path.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG_SFT["default"]["seed"],
        help="Random seed.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    wandb.init(
        project=args.project,
        name=args.run_name,
        config=CONFIG_SFT,
    )

    try:
        log.info("Loading model...")
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
        trainer.train()

        log.info("Training complete.")

        push_hub(
            name=args.run_name,
            trainer=trainer,
        )

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()