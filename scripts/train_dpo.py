import argparse
import logging

import wandb
from datasets import load_dataset

from src.models.huggingface import build_model
from src.peft.config import get_lora_config
from src.rlhf.dpo.formatting import convert_to_conversational_dpo_format
from src.rlhf.dpo.trainer import build_dpo_trainer

from src.utils.config import load_config
from src.utils.hub import push_hub
from src.utils.seed import set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

log = logging.getLogger(__name__)

CONFIG_DPO = load_config("configs/rlhf/dpo.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Direct Preference Optimization (DPO) model.",
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
        default=CONFIG_DPO["output"]["hub_name"],
        help="Weights & Biases run name / Hugging Face Hub repository name.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=CONFIG_DPO["data"]["path"],
        help="Hugging Face dataset name or path.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG_DPO["default"]["seed"],
        help="Random seed.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    wandb.init(
        project=args.project,
        name=args.run_name,
        config=CONFIG_DPO,
    )

    try:
        log.info("Loading model...")
        model, tokenizer = build_model(args.model)

        log.info("Loading dataset: %s", args.dataset)
        dataset = load_dataset(args.dataset)
        dataset = dataset.map(convert_to_conversational_dpo_format)

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
        trainer.train()
        
        log.info("Training complete.")
        
        model.save_pretrained(CONFIG_DPO["output"]["dir"])
        tokenizer.save_pretrained(CONFIG_DPO["output"]["dir"])

        push_hub(
            name=args.run_name,
            trainer=trainer,
        )

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()