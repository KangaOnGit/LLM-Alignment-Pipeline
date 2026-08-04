from src.utils.config import load_config
from peft import LoraConfig

def get_lora_config(**overrides) -> LoraConfig:
    CONFIG = load_config("configs/peft/lora.yaml")

    config = {
        "r": CONFIG["default"]["r"],
        "lora_alpha": CONFIG["default"]["alpha"],
        "lora_dropout": CONFIG["default"]["dropout"],
        "bias": CONFIG["default"]["bias"],
        "task_type": CONFIG["default"]["task_type"],
        "target_modules": CONFIG["target_modules"],
    }

    config.update(overrides)

    return LoraConfig(**config)