import logging

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.rlhf.ppo.base import PPOModels

log = logging.getLogger(__name__)


def build_policy_and_reward_models(
    policy_model_name: str = "gpt2",
    reward_model_name: str = "gpt2",
) -> PPOModels:
    """Load the tokenizer, policy model, reference policy, and reward model."""

    log.info("Loading tokenizer: %s", policy_model_name)
    tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading policy model: %s", policy_model_name)
    policy = AutoModelForCausalLM.from_pretrained(policy_model_name)
    policy.train()

    log.info("Loading reference policy: %s", policy_model_name)
    reference_policy = AutoModelForCausalLM.from_pretrained(policy_model_name)
    reference_policy.eval()

    log.info("Loading reward model: %s", reward_model_name)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name,
        num_labels=1,
    )
    reward_model.eval()

    return PPOModels(
        tokenizer=tokenizer,
        policy=policy,
        reference_policy=reference_policy,
        reward_model=reward_model,
    )