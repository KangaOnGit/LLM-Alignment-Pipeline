import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer


class GPT2RewardModel(nn.Module):
    def __init__(self, model_name: str = "gpt2"):
        super().__init__()
        self.transformer = GPT2Model.from_pretrained(model_name)
        self.value_head = nn.Linear(self.transformer.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        value = self.value_head(last_hidden[:, -1, :])
        return value.squeeze(-1)


def build_tokenizer(model_name: str = "gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode_batch(samples, tokenizer):
    chosen_texts = [sample["prompt"] + " " + sample["chosen"] for sample in samples]
    rejected_texts = [sample["prompt"] + " " + sample["rejected"] for sample in samples]

    chosen = tokenizer(chosen_texts, padding=True, truncation=True, return_tensors="pt")
    rejected = tokenizer(rejected_texts, padding=True, truncation=True, return_tensors="pt")
    return chosen, rejected


def train_reward_model(model, tokenizer, samples, epochs: int = 200, learning_rate: float = 2e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MarginRankingLoss(margin=1.0)

    chosen, rejected = encode_batch(samples, tokenizer)
    chosen = {key: value for key, value in chosen.items()}
    rejected = {key: value for key, value in rejected.items()}

    for epoch in range(epochs):
        model.train()
        r_chosen = model(**chosen)
        r_rejected = model(**rejected)
        target = torch.ones_like(r_chosen)
        loss = loss_fn(r_chosen, r_rejected, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    return model
