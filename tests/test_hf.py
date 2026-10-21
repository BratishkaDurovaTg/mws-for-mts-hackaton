import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from compactfacts.models.hf import (
    HFExtractor,
    LocalModelError,
    context_limit,
    load_local_runtime,
    local_snapshot_metadata,
)
from compactfacts.schemas import InferenceConfig, Message, Ticket


class Tensor:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return Tensor(self.values[index])

    def tolist(self):
        return self.values

    def to(self, _device):
        return self


class Torch:
    long = "long"
    float32 = "float32"
    cuda = SimpleNamespace(is_available=lambda: False)

    def manual_seed(self, seed):
        self.seed = seed

    def tensor(self, values, **_kwargs):
        return Tensor(values)

    def ones_like(self, tensor):
        return Tensor([[1 for _ in row] for row in tensor.values])

    def inference_mode(self):
        return nullcontext()


class Tokenizer:
    chat_template = "fake native chat template"
    model_max_length = 131072
    pad_token_id = eos_token_id = 9
    bos_token_id = None

    def __init__(self):
        self.prompts = []
        self.output_text = ""
        self.decoded_ids = None

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["truncation"] is False
        self.prompts.append(messages)
        return {"input_ids": [1, 2]}

    def decode(self, ids, **_kwargs):
        self.decoded_ids = ids
        return self.output_text


class Model:
    config = SimpleNamespace(max_position_embeddings=40960, model_type="qwen3")

    def __init__(self, tokenizer, responses=None):
        self.tokenizer = tokenizer
        self.responses = iter(responses or [([3, 9], '{"facts":[]}')])
        self.calls = []

    def eval(self):
        return self

    def to(self, _device):
        return self

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        ids, self.tokenizer.output_text = next(self.responses)
        return Tensor([kwargs["input_ids"].values[0] + ids])


def config(**updates):
    return InferenceConfig(
        backend="hf",
        model_path="models/local",
        model_id="test/model",
        revision="a" * 40,
        max_new_tokens=8,
        **updates,
    )


def loaded(monkeypatch, responses=None, **updates):
    tokenizer = Tokenizer()
    model = Model(tokenizer, responses)
    torch = Torch()
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(GenerationConfig=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    return HFExtractor.from_loaded(config(**updates), model, tokenizer, torch), model, tokenizer


def ticket():
    return Ticket(id="t", messages=[Message(id="u", role="user", content="No issue")], facts=[])


def test_hf_explicit_greedy_settings_and_output_prefix_removed(monkeypatch):
    extractor, model, tokenizer = loaded(monkeypatch)
    prediction = extractor.predict(ticket())
    assert prediction.status == "ok" and prediction.facts == []
    assert prediction.generated_tokens == 2 and prediction.attempts == 1
    generation = model.calls[0]["generation_config"]
    assert generation.do_sample is False and generation.num_beams == 1
    assert generation.max_new_tokens == 8
    assert tokenizer.decoded_ids == [3, 9]


def test_hf_retries_bounded_and_counts_tokens_from_failed_attempt(monkeypatch):
    extractor, model, tokenizer = loaded(
        monkeypatch, [([3, 9], "bad JSON"), ([4, 9], '{"facts":[]}')], max_retries=1
    )
    prediction = extractor.predict(ticket())
    assert prediction.status == "ok" and prediction.attempts == 2
    assert prediction.generated_tokens == 4 and len(model.calls) == 2
    assert "invalid_json" in tokenizer.prompts[1][0]["content"]
    assert "bad JSON" not in tokenizer.prompts[1][0]["content"]


def test_exhausted_retry_is_visible_failure_not_empty_success(monkeypatch):
    extractor, model, _ = loaded(monkeypatch, [([3, 9], "invalid")] * 3, max_retries=2)
    prediction = extractor.predict(ticket())
    assert prediction.status == "error" and prediction.error_code == "invalid_json"
    assert prediction.attempts == 3 and prediction.generated_tokens == 6
    assert len(model.calls) == 3


def test_hf_refuses_long_token_input_without_generation(monkeypatch):
    extractor, model, _ = loaded(monkeypatch, max_input_tokens=1)
    assert extractor.predict(ticket()).error_code == "input_too_long"
    assert model.calls == []


def test_hf_refuses_context_overflow_with_short_prompt(monkeypatch):
    extractor, model, _ = loaded(monkeypatch)
    model.config = SimpleNamespace(max_position_embeddings=9, model_type="qwen3")
    assert extractor.predict(ticket()).error_code == "input_too_long"
    assert model.calls == []


def test_generation_limit_does_not_parse_truncated_json_as_success(monkeypatch):
    extractor, _, _ = loaded(monkeypatch, [([3] * 8, '{"facts":[]}')])
    prediction = extractor.predict(ticket())
    assert prediction.error_code == "output_limit" and prediction.status == "error"
    assert prediction.generated_tokens == 8


def test_missing_model_is_lazy_typed_failure(tmp_path):
    extractor = HFExtractor(config().model_copy(update={"model_path": str(tmp_path / "missing")}))
    assert extractor.predict(ticket()).error_code == "local_model_missing"


def test_qwen3_context_uses_conservative_card_not_tokenizer_advertisement():
    assert context_limit(Model.config, Tokenizer()) == 32768
    with pytest.raises(LocalModelError, match="unknown_context"):
        context_limit(SimpleNamespace(), Tokenizer())


def test_loader_is_local_only_native_and_safetensors(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"qwen3"}')
    (tmp_path / "tokenizer_config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"fake-test-only")
    tokenizer = Tokenizer()
    model = Model(tokenizer)
    calls = []

    def get_tokenizer(path, **kwargs):
        calls.append(("tokenizer", path, kwargs))
        return tokenizer

    def get_model(path, **kwargs):
        calls.append(("model", path, kwargs))
        return model

    monkeypatch.setitem(sys.modules, "torch", Torch())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=get_tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=get_model),
        ),
    )
    _, _, _, metadata = load_local_runtime(
        config().model_copy(update={"model_path": str(tmp_path)})
    )
    assert metadata["weight_bytes"] == 14
    assert metadata["weight_content_hashes_verified"] is False
    for _, path, kwargs in calls:
        assert path == str(tmp_path)
        assert kwargs["local_files_only"] is True and kwargs["trust_remote_code"] is False
        assert kwargs["revision"] == "a" * 40
    assert calls[1][2]["use_safetensors"] is True
    assert calls[1][2]["dtype"] == "float32"


def test_weight_index_cannot_escape_model_directory(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"fake-test-only")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "../other.safetensors"}})
    )
    with pytest.raises(LocalModelError, match="invalid_weight_index"):
        local_snapshot_metadata(str(tmp_path))
