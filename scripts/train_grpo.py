import argparse
import logging

import wandb
from datasets import load_dataset

from src.models.unsloth import build_fast_model
from src.peft.unsloth import apply_lora

from src.rlhf.grpo.reward.answer import (
    check_answer,
    check_numbers,
)
from src.rlhf.grpo.reward.match_format import (
    match_format_exactly,
    match_format_approximately,
)
from src.rlhf.grpo.trainer import build_grpo_trainer

from src.utils.config import load_config
from src.utils.hub import push_hub
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

log = logging.getLogger(__name__)

CONFIG_GRPO = load_config("configs/rlhf/grpo.yaml")

DEFAULT_REWARD_FUNCS = [
    match_format_exactly,
    match_format_approximately,
    check_answer,
    check_numbers,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GRPO model.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=CONFIG_GRPO["default"]["model"],
        help="Model name for GRPO.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=CONFIG_GRPO["data"]["path"],
        help="Hugging Face dataset name or path.",
    )

    parser.add_argument(
        "--project",
        type=str,
        default=CONFIG_GRPO["wandb"]["project"],
        help="Weights & Biases project name.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="llama-3.2-3b-grpo",
        help="Weights & Biases run name.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG_GRPO["default"]["seed"],
        help="Random seed.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    wandb.init(
        project=args.project,
        name=args.run_name,
        config=CONFIG_GRPO,
    )

    try:
        log.info("Loading model...")
        model, tokenizer = build_fast_model(args.model)

        log.info("Applying LoRA...")
        model = apply_lora(model)

        log.info("Loading dataset: %s", args.dataset)
        dataset = load_dataset(args.dataset)["train"]

        log.info("Building trainer...")
        trainer = build_grpo_trainer(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            reward_funcs=DEFAULT_REWARD_FUNCS,
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