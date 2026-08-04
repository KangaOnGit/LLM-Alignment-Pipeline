import logging
import torch

from datasets import load_dataset
from unsloth import FastLanguageModel
from vllm import SamplingParams

from src.rlhf.grpo.formatting import convert_to_conversational_grpo_format
from src.utils.config import load_config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

log = logging.getLogger(__name__)

CONFIG_GRPO = load_config("configs/rlhf/grpo.yaml")


def main():

    model_name = CONFIG_GRPO["default"]["model"]
    lora_path = CONFIG_GRPO["output"]["dir"]

    log.info("Loading base model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=64,
        gpu_memory_utilization=0.8,
    )

    FastLanguageModel.for_inference(model)


    log.info("Loading dataset...")

    raw_dataset = load_dataset(
        CONFIG_GRPO["data"]["path"],
        split="train",
    )

    dataset = convert_to_conversational_grpo_format(
        raw_dataset,
        limit=8000,
    )


    idx = 0

    messages = dataset[idx]["prompt"]

    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    sampling_params = SamplingParams(
        temperature=CONFIG_GRPO["generation"]["temperature"],
        top_p=CONFIG_GRPO["generation"]["top_p"],
        max_tokens=CONFIG_GRPO["train"]["max_completion_length"],
    )

    log.info("Generating without LoRA...")

    output_base = model.fast_generate(
        [text],
        sampling_params=sampling_params,
        lora_request=None,
    )[0].outputs[0].text


    log.info("Generating with LoRA...")

    lora_request = model.load_lora(lora_path)

    output_lora = model.fast_generate(
        [text],
        sampling_params=sampling_params,
        lora_request=lora_request,
    )[0].outputs[0].text


    log.info("=" * 80)
    log.info("Problem:")
    log.info(dataset[idx]["prompt"][-1]["content"])

    log.info("\nBase Model Response:")
    log.info(output_base)

    log.info("\nGRPO LoRA Response:")
    log.info(output_lora)

    log.info("\nGround Truth Answer:")
    log.info(dataset[idx]["answer"])


if __name__ == "__main__":
    main()