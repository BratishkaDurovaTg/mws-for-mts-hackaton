"""Lazy, local-only Transformers extraction; no checkpoint or remote-code downloads."""

from pathlib import Path

from compactfacts.inference import error_prediction, parse_generation
from compactfacts.models.prompts import build_messages, template_tokens
from compactfacts.schemas import InferenceConfig, Prediction, Ticket
from compactfacts.utils.io import file_hash, strict_loads


class LocalModelError(ValueError):
    """Safe aggregate setup error; never embeds a source ticket or credentials."""


def local_snapshot_metadata(path: str) -> dict:
    """Validate required local files and record hashes without reading GB of weights."""
    root = Path(path)
    if not root.is_dir():
        raise LocalModelError("local_model_missing")
    if (root / "adapter_config.json").exists():
        raise LocalModelError("base_path_is_adapter")
    for name in ("config.json", "tokenizer_config.json"):
        if not (root / name).is_file():
            raise LocalModelError("local_model_incomplete")
    raw = strict_loads((root / "config.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("quantization_config") is not None:
        raise LocalModelError("unsupported_model_config")
    weights = sorted(root.glob("*.safetensors"))
    if not weights:
        raise LocalModelError("safetensors_required")
    index = root / "model.safetensors.index.json"
    if index.exists():
        shard_map = strict_loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        if not shard_map or any(
            Path(name).name != name
            or not (root / name).is_file()
            or not name.endswith(".safetensors")
            for name in shard_map.values()
        ):
            raise LocalModelError("invalid_weight_index")
    small_files = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "generation_config.json",
        "model.safetensors.index.json",
    ]
    return {
        "small_file_sha256": {
            name: file_hash(root / name) for name in small_files if (root / name).is_file()
        },
        "weight_files": [
            {"name": weight.name, "bytes": weight.stat().st_size} for weight in weights
        ],
        "weight_bytes": sum(weight.stat().st_size for weight in weights),
        "weight_content_hashes_verified": False,
        "provenance_note": (
            "configured model_id/revision are provenance claims; local weights not byte-verified"
        ),
    }


def context_limit(model_config, tokenizer) -> int:
    limit = getattr(model_config, "max_position_embeddings", None)
    if type(limit) is not int or limit <= 0:
        raise LocalModelError("unknown_context_limit")
    if getattr(model_config, "model_type", None) == "qwen3":
        limit = min(limit, 32768)  # Pinned model cards vs larger config/tokenizer values.
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if type(tokenizer_limit) is int and 0 < tokenizer_limit < 10**9:
        limit = min(limit, tokenizer_limit)
    return limit


def load_local_runtime(config: InferenceConfig):
    """Shared inference/training loader; every from_pretrained is local-only."""
    metadata = local_snapshot_metadata(config.model_path)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if config.device == "cuda" and not torch.cuda.is_available():
        raise LocalModelError("device_unavailable")
    if (
        config.device == "cuda"
        and config.dtype == "bfloat16"
        and (not torch.cuda.is_bf16_supported())
    ):
        raise LocalModelError("precision_unsupported")
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise LocalModelError("device_unavailable")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        local_files_only=True,
        trust_remote_code=False,
        revision=config.revision,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise LocalModelError("chat_template_required")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        local_files_only=True,
        trust_remote_code=False,
        revision=config.revision,
        use_safetensors=True,
        dtype=getattr(torch, config.dtype),
    )
    model.to(config.device)
    model.eval()
    context_limit(model.config, tokenizer)
    return model, tokenizer, torch, metadata


class HFExtractor:
    def __init__(self, config: InferenceConfig):
        if config.backend != "hf":
            raise ValueError("HFExtractor requires backend=hf")
        self.config = config
        self._model = self._tokenizer = self._torch = None
        self._load_error = None
        self.metadata = None

    @classmethod
    def from_loaded(cls, config, model, tokenizer, torch_module):
        """Reuse a model for generation checks during a local training run."""
        extractor = cls(config)
        extractor._model, extractor._tokenizer, extractor._torch = model, tokenizer, torch_module
        return extractor

    def _ensure_loaded(self):
        if self._model is not None or self._load_error is not None:
            return
        try:
            loaded = load_local_runtime(self.config)
            self._model, self._tokenizer, self._torch, self.metadata = loaded
            if self.config.adapter_path:
                adapter = Path(self.config.adapter_path)
                if (
                    not (adapter / "adapter_config.json").is_file()
                    or not (adapter / "adapter_model.safetensors").is_file()
                ):
                    raise LocalModelError("local_adapter_incomplete")
                from peft import PeftModel

                self._model = PeftModel.from_pretrained(
                    self._model,
                    str(adapter),
                    local_files_only=True,
                    is_trainable=False,
                )
                self._model.to(self.config.device)
                self._model.eval()
        except ImportError:
            self._model = None
            self._load_error = "ml_dependency_missing"
        except LocalModelError as exc:
            self._model = None
            self._load_error = str(exc)
        except Exception:
            self._model = None
            self._load_error = "model_load_failed"

    def prepare(self) -> bool:
        """Materialize the local runtime so benchmarks can separate startup from warm calls."""
        self._ensure_loaded()
        return self._load_error is None

    def predict(self, ticket: Ticket) -> Prediction:
        if sum(len(message.content) for message in ticket.messages) > self.config.max_input_chars:
            return error_prediction(ticket.id, "input_too_long")
        self._ensure_loaded()
        if self._load_error:
            return error_prediction(ticket.id, self._load_error)
        total_tokens = 0
        previous_error = None
        for attempt in range(1, self.config.max_retries + 2):
            try:
                messages = build_messages(ticket, self.config.allowed_types)
                if previous_error is not None:
                    messages[0]["content"] += (
                        "\nValidation retry: "
                        "The previous attempt failed validation (" + previous_error + "). "
                        "Return complete JSON conforming to the same schema and evidence rules."
                    )
                ids = template_tokens(self._tokenizer, messages, generation_prompt=True)
                if len(ids) > self.config.max_input_tokens or (
                    len(ids) + self.config.max_new_tokens
                    > context_limit(self._model.config, self._tokenizer)
                ):
                    return error_prediction(ticket.id, "input_too_long", total_tokens).model_copy(
                        update={"attempts": attempt}
                    )
                self._torch.manual_seed(self.config.seed)
                inputs = self._torch.tensor([ids], dtype=self._torch.long).to(self.config.device)
                attention = self._torch.ones_like(inputs)
                from transformers import GenerationConfig

                generation = GenerationConfig(
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=self.config.max_new_tokens,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    bos_token_id=self._tokenizer.bos_token_id,
                    use_cache=True,
                )
                self._model.eval()
                with self._torch.inference_mode():
                    result = self._model.generate(
                        input_ids=inputs,
                        attention_mask=attention,
                        generation_config=generation,
                    )
                output = result[0].tolist()[len(ids) :]
                total_tokens += len(output)
                eos = self._tokenizer.eos_token_id
                if len(output) >= self.config.max_new_tokens and (not output or output[-1] != eos):
                    prediction = error_prediction(ticket.id, "output_limit", total_tokens)
                else:
                    text = self._tokenizer.decode(output, skip_special_tokens=True)
                    prediction = parse_generation(ticket, text, self.config, total_tokens)
                prediction = prediction.model_copy(update={"attempts": attempt})
                if prediction.status == "ok":
                    return prediction
                previous_error = prediction.error_code
            except Exception:
                return error_prediction(ticket.id, "generation_failed", total_tokens).model_copy(
                    update={"attempts": attempt}
                )
        return prediction
