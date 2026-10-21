"""Response-only SFT features and position-based padding, without mandatory torch."""

from compactfacts.models.prompts import build_messages, target_json, template_tokens
from compactfacts.schemas import Ticket


def build_features(
    ticket: Ticket, tokenizer, max_length: int, allowed_types: list[str] | None = None
) -> dict[str, list[int]]:
    if type(max_length) is not int or max_length <= 0:
        raise ValueError("max_length must be positive")
    messages = build_messages(ticket, allowed_types)
    target = target_json(ticket)
    prompt = template_tokens(tokenizer, messages, generation_prompt=True)
    complete = template_tokens(
        tokenizer, messages + [{"role": "assistant", "content": target}], generation_prompt=False
    )
    if complete[: len(prompt)] != prompt:
        raise ValueError("chat_template_token_prefix_mismatch")
    if len(complete) <= len(prompt):
        raise ValueError("no_supervised_response")
    eos = getattr(tokenizer, "eos_token_id", None)
    if type(eos) is not int:
        raise ValueError("eos_token_required")
    if eos not in complete[len(prompt) :]:
        complete = complete + [eos]
    if len(complete) > max_length:
        raise ValueError("training_sequence_too_long; no truncation applied")
    return {
        "input_ids": complete,
        "attention_mask": [1] * len(complete),
        "labels": [-100] * len(prompt) + complete[len(prompt) :],
    }


class ResponseCollator:
    def __init__(self, pad_token_id: int, *, tensorize: bool = False):
        if type(pad_token_id) is not int:
            raise ValueError("pad_token_required")
        self.pad_token_id = pad_token_id
        self.tensorize = tensorize

    def __call__(self, features: list[dict]) -> dict:
        if not features:
            raise ValueError("cannot collate empty batch")
        for feature in features:
            n = len(feature["input_ids"])
            if not n or any(len(feature[key]) != n for key in ("attention_mask", "labels")):
                raise ValueError("inconsistent feature lengths")
            if all(label == -100 for label in feature["labels"]):
                raise ValueError("no_supervised_response")
        width = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = width - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            # Mask positions that we added, not every occurrence of the PAD/EOS token value.
            batch["labels"].append(feature["labels"] + [-100] * padding)
        if self.tensorize:
            import torch

            return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
        return batch
