import argparse
import logging

import wandb

from src.sft.datasets import load_dataset
from src.utils.common.build_model import build_model, build_tokenizer
from src.sft.trainer import build_sft_trainer
from src.utils.config import HF_TOKEN, load_config, push_hub, get_qlora_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

CONFIG_SFT = load_config("configs/sft.yaml")
CONFIG_QLORA = load_config("configs/qlora.yaml")

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
        "--config",
        type=str,
        default="configs/sft.yaml",
        help="Path to the SFT configuration file.",
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
        default=CONFIG_SFT["data"]["path_sft"],
        help="Weights & Biases run name.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)

    wandb.init(
        project=args.project,
        name=args.run_name,
        config=config,
    )

    model = build_model(args.model)

    tokenizer = build_tokenizer(args.model)

    log.info("Loading dataset: %s", args.dataset)
    dataset = load_dataset(args.dataset)

    log.info("Building LoRA configuration...")
    peft_config = get_qlora_config(
        r=CONFIG_QLORA["default"]["r"],
        lora_alpha=CONFIG_QLORA["default"]["alpha"],
        lora_dropout=CONFIG_QLORA["default"]["dropout"],
        bias=CONFIG_QLORA["default"]["bias"],
        task_type=CONFIG_QLORA["default"]["task_type"],
        target_modules=CONFIG_QLORA["target_modules"]
    )

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