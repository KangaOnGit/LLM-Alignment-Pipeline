def apply_lora(
    model,
    **overrides,
):
    CONFIG = load_config("configs/peft/lora.yaml")

    config = {
        "r": CONFIG["default"]["r"],
        "lora_alpha": CONFIG["default"]["alpha"],
        "lora_dropout": CONFIG["default"]["dropout"],
        "bias": CONFIG["default"]["bias"],
        "target_modules": CONFIG["target_modules"],
        "use_gradient_checkpointing": "unsloth",
        "random_state": 3407,
    }

    config.update(overrides)

    return FastLanguageModel.get_peft_model(
        model,
        **config,
    )