from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
)

@dataclass
class PPOModels:
    tokenizer: PreTrainedTokenizer
    policy: AutoModelForCausalLM
    reference_policy: AutoModelForCausalLM
    reward_model: AutoModelForSequenceClassification