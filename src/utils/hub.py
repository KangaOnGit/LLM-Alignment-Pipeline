from transformers import Trainer

from src.utils.config import HF_TOKEN

def push_hub(
    name: str,
    trainer: Trainer,
) -> None:
    trainer.push_to_hub(name, token=HF_TOKEN)