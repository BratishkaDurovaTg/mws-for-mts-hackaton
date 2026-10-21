"""Explicit local LoRA stages with persisted smoke/tiny-overfit gates.

No training starts on import. Real model training remains unverified until run
with user-provided local weights and a frozen, reviewed data split.
"""

import math
import os
import platform
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from compactfacts.evaluation import evaluate
from compactfacts.inference import parse_generation
from compactfacts.models.hf import HFExtractor, context_limit, load_local_runtime
from compactfacts.models.prompts import build_messages, target_json
from compactfacts.postprocessing import fact_key
from compactfacts.schemas import InferenceConfig, NonEmpty, StrictModel, Ticket
from compactfacts.training.features import ResponseCollator, build_features
from compactfacts.utils.io import file_hash, stable_hash, strict_loads, write_json
from compactfacts.utils.provenance import provenance


class TrainingConfig(StrictModel):
    inference: InferenceConfig
    stage: Literal["smoke", "tiny_overfit", "full"] = "smoke"
    output_dir: NonEmpty
    split_manifest_path: NonEmpty
    prior_gate_manifest: str | None = None
    baseline_run_path: str | None = None
    data_origin: Literal["synthetic", "provisional", "official"] = "synthetic"
    max_length: int = Field(default=4096, gt=1)
    batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=4, gt=0)
    learning_rate: float = Field(default=0.0002, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    optimizer: Literal["adamw_torch"] = "adamw_torch"
    scheduler: Literal["linear", "cosine", "constant"] = "linear"
    warmup_steps: int = Field(default=0, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    lora_rank: int = Field(default=8, gt=0)
    lora_alpha: int = Field(default=16, gt=0)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1, allow_inf_nan=False)
    target_modules: list[NonEmpty] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    gradient_checkpointing: bool = True
    tiny_examples: int = Field(default=2, ge=1, le=16)
    smoke_steps: int = Field(default=2, ge=1, le=10)
    tiny_steps: int = Field(default=100, ge=2, le=1000)
    full_steps: int = Field(default=400, ge=1)
    save_steps: int = Field(default=50, ge=1)
    minimum_loss_ratio: float = Field(default=0.95, gt=0, lt=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validated_stage(self) -> Self:
        if self.inference.backend != "hf" or self.inference.adapter_path is not None:
            raise ValueError("training requires pinned local HF base without an existing adapter")
        if self.inference.parser_mode != "strict" or self.inference.max_retries:
            raise ValueError("training generation checks require strict parser and zero retries")
        if not self.target_modules or len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("target_modules must be non-empty and unique")
        if self.stage != "smoke" and self.prior_gate_manifest is None:
            raise ValueError("later stages require a persisted prior gate manifest")
        if self.stage == "full" and self.data_origin == "synthetic":
            raise ValueError("full training on synthetic fixtures is disabled")
        if self.stage == "full" and self.baseline_run_path is None:
            raise ValueError("full training requires a measured prompt-only baseline run")
        return self


def validate_split(train: list[Ticket], dev: list[Ticket], manifest_path: str) -> dict:
    if not train or not dev or any(ticket.facts is None for ticket in train + dev):
        raise ValueError("training requires non-empty annotated train and dev splits")
    ids = [ticket.id for ticket in train + dev]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate or overlapping train/dev IDs")
    manifest = strict_loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "compactfacts.split.v1":
        raise ValueError("unsupported frozen split manifest")
    digest = manifest.get("manifest_hash")
    if digest != stable_hash({k: v for k, v in manifest.items() if k != "manifest_hash"}):
        raise ValueError("split manifest hash mismatch")
    if set(manifest.get("train_ids", [])) != {ticket.id for ticket in train} or set(
        manifest.get("dev_ids", [])
    ) != {ticket.id for ticket in dev}:
        raise ValueError("data IDs do not match frozen split")
    expected = manifest.get("ticket_hashes", {})
    if any(
        expected.get(ticket.id) != stable_hash(ticket.model_dump(mode="json"))
        for ticket in train + dev
    ):
        raise ValueError("data content or annotations changed since split freeze")
    if manifest.get("leakage_audit", {}).get("passed") is not True:
        raise ValueError("frozen split has not passed its declared leakage screen")
    return manifest


def training_fingerprint(
    config: TrainingConfig,
    train: list[Ticket],
    dev: list[Ticket],
    snapshot_metadata: dict,
    split_manifest: dict,
) -> str:
    recipe = config.model_dump(
        mode="json",
        exclude={
            "stage",
            "output_dir",
            "prior_gate_manifest",
            "split_manifest_path",
            "baseline_run_path",
        },
    )
    # Absolute/relative path spelling is not model identity. Local file hashes/sizes are recorded.
    recipe["inference"].pop("model_path", None)
    return stable_hash(
        {
            "recipe": recipe,
            "train": [
                ticket.model_dump(mode="json") for ticket in sorted(train, key=lambda t: t.id)
            ],
            "dev": [ticket.model_dump(mode="json") for ticket in sorted(dev, key=lambda t: t.id)],
            "snapshot": snapshot_metadata,
            "split_manifest_hash": split_manifest["manifest_hash"],
            "prompt_hash": stable_hash(
                build_messages(Ticket(id="contract", messages=[]), config.inference.allowed_types)[
                    0
                ]
            ),
            "pipeline_code_sha256": {
                name: file_hash(Path(__file__).parents[1] / name)
                for name in (
                    "training/runner.py",
                    "training/features.py",
                    "models/prompts.py",
                    "models/hf.py",
                    "schemas.py",
                    "inference/__init__.py",
                    "inference/parser.py",
                    "inference/audit.py",
                    "postprocessing/__init__.py",
                )
            },
        }
    )


def validate_prior_gate(config: TrainingConfig, fingerprint: str) -> dict:
    if config.stage == "smoke":
        return {}
    prior = strict_loads(Path(config.prior_gate_manifest).read_text(encoding="utf-8"))
    if prior.get("schema_version") != "compactfacts.training.v1" or (
        prior.get("fingerprint") != fingerprint or prior.get("status") != "passed"
    ):
        raise ValueError("prior gate is failed, incompatible, or belongs to changed data/config")
    gates = prior.get("gates", {})
    required = ["smoke"] if config.stage == "tiny_overfit" else ["smoke", "tiny_overfit"]
    if any(gates.get(stage, {}).get("passed") is not True for stage in required):
        raise ValueError("required smoke/tiny-overfit gate is missing")
    if gates["smoke"].get("finite_updates") is not True:
        raise ValueError("smoke gate lacks finite updates evidence")
    if config.stage == "full" and any(
        gates["tiny_overfit"].get(check) is not True
        for check in ("finite_updates", "loss_decreased", "generation_exact")
    ):
        raise ValueError("tiny gate lacks decreasing loss or exact generated facts evidence")
    return {stage: gates[stage] for stage in required}


def _git_state() -> dict:
    state = provenance()
    return {
        "commit": state["git_commit"],
        "dirty": state["git_dirty"],
        "package_source_sha256": state["package_source_sha256"],
    }


def validate_baseline(
    config: TrainingConfig,
    train: list[Ticket],
    dev: list[Ticket],
    weight_sha256: dict | None = None,
) -> dict | None:
    """Full training follows an actual prompt-only run on these frozen dev records."""
    if config.stage != "full":
        return None
    if any(ticket.metadata.get("synthetic") is True for ticket in train + dev):
        raise ValueError("full training refuses actual synthetic records regardless of data_origin")
    baseline = strict_loads(Path(config.baseline_run_path).read_text(encoding="utf-8"))
    expected_dev = stable_hash(
        [ticket.model_dump(mode="json") for ticket in sorted(dev, key=lambda ticket: ticket.id)]
    )
    if (
        baseline.get("canonical_data_sha256") != expected_dev
        or baseline.get("base_model") != config.inference.model_id
        or baseline.get("revision") != config.inference.revision
    ):
        raise ValueError("baseline data/base/revision do not match the frozen training comparison")
    baseline_config = baseline.get("config", {})
    if baseline_config.get("backend") != "hf" or baseline_config.get("adapter_path") is not None:
        raise ValueError("baseline must use the prompt-only HF base without an adapter")
    for key in (
        "allowed_types",
        "parser_mode",
        "require_evidence",
        "enable_thinking",
        "max_input_tokens",
        "max_new_tokens",
        "seed",
        "max_retries",
    ):
        if baseline_config.get(key) != getattr(config.inference, key):
            raise ValueError("baseline extraction contract/decoding differs from training checks")
    primary = baseline.get("metrics", {}).get("primary", {})
    if (
        baseline.get("metrics", {}).get("technical_failure_rate") != 0
        or baseline.get("benchmark", {}).get("measurement_status") != "measured"
    ):
        raise ValueError(
            "baseline must have successful measured inference without technical failures"
        )
    if (
        not isinstance(primary.get("name"), str)
        or not primary["name"].strip()
        or type(primary.get("score")) not in {int, float}
        or not math.isfinite(primary["score"])
    ):
        raise ValueError("baseline lacks measured finite primary metric")
    baseline_weights = baseline.get("model_weight_sha256")
    if not isinstance(baseline_weights, dict) or not baseline_weights:
        raise ValueError("baseline lacks local weight content hashes")
    if weight_sha256 is not None and baseline_weights != weight_sha256:
        raise ValueError("baseline weight content differs from the current local base")
    return {
        "file_sha256": file_hash(config.baseline_run_path),
        "primary": primary,
        "canonical_dev_sha256": expected_dev,
        "run_id": baseline.get("id"),
        "model_weight_sha256": baseline_weights,
    }


def _environment(torch, config: TrainingConfig) -> dict:
    packages = {}
    for package in ("torch", "transformers", "peft", "accelerate", "safetensors"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "packages": packages,
        "device": config.inference.device,
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "world_size": 1,
    }


def _generation_checks(model, tokenizer, torch, config, tickets) -> tuple[list, list]:
    extractor = HFExtractor.from_loaded(config.inference, model, tokenizer, torch)
    predictions = [extractor.predict(ticket) for ticket in tickets]
    checks = [
        {
            "example_index": index,
            "status": prediction.status,
            "error_code": prediction.error_code,
            "generated_tokens": prediction.generated_tokens,
            "exact_fact_set": prediction.status == "ok"
            and (
                {fact_key(fact) for fact in prediction.facts}
                == {fact_key(fact) for fact in ticket.facts}
            ),
        }
        for index, (ticket, prediction) in enumerate(zip(tickets, predictions, strict=True))
    ]
    return checks, predictions


def run_training(
    config: TrainingConfig,
    train_tickets: list[Ticket],
    dev_tickets: list[Ticket],
    official_scorer=None,
) -> dict:
    """Run exactly one explicit stage; no automatic escalation into full training."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("this initial runner supports a single process only")
    train_tickets = sorted(train_tickets, key=lambda ticket: ticket.id)
    dev_tickets = sorted(dev_tickets, key=lambda ticket: ticket.id)
    split = validate_split(train_tickets, dev_tickets, config.split_manifest_path)
    baseline_evidence = validate_baseline(config, train_tickets, dev_tickets)
    for ticket in train_tickets + dev_tickets:
        if parse_generation(ticket, target_json(ticket), config.inference).status != "ok":
            raise ValueError("gold target violates configured types/evidence contract")
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("training output must be a new or empty directory")
    # Dependencies and local weight files are intentionally required only after pure preflight.
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(config.inference.seed)
    model, tokenizer, torch, snapshot = load_local_runtime(config.inference)
    hash_started = time.perf_counter()
    snapshot["weight_sha256"] = {
        weight["name"]: file_hash(Path(config.inference.model_path) / weight["name"])
        for weight in snapshot["weight_files"]
    }
    weight_hash_seconds = time.perf_counter() - hash_started
    snapshot["weight_content_hashes_verified"] = True
    snapshot["weight_hash_verification_scope"] = (
        "local content hashing only; not compared with official Hub digests"
    )
    snapshot["provenance_note"] = (
        "local content hashes computed; model_id/revision not verified against Hub digests"
    )
    if config.stage == "full":
        baseline_evidence = validate_baseline(
            config, train_tickets, dev_tickets, snapshot["weight_sha256"]
        )
    fingerprint = training_fingerprint(config, train_tickets, dev_tickets, snapshot, split)
    gates = validate_prior_gate(config, fingerprint)
    if config.max_length > context_limit(model.config, tokenizer):
        raise ValueError("training max_length exceeds conservative model context")
    train_features = [
        build_features(ticket, tokenizer, config.max_length, config.inference.allowed_types)
        for ticket in train_tickets
    ]
    dev_features = [
        build_features(ticket, tokenizer, config.max_length, config.inference.allowed_types)
        for ticket in dev_tickets
    ]
    n = min(config.tiny_examples, len(train_tickets))
    stage_train = train_features if config.stage == "full" else train_features[:n]
    stage_eval = dev_features if config.stage == "full" else train_features[:n]
    steps = {
        "smoke": config.smoke_steps,
        "tiny_overfit": config.tiny_steps,
        "full": config.full_steps,
    }[config.stage]
    if config.warmup_steps >= steps:
        raise ValueError("warmup_steps must be smaller than this stage's optimizer steps")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none",
        ),
    )
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable == 0:
        raise ValueError("LoRA injection produced no trainable parameters")
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.enable_input_require_grads()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    args = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        max_steps=steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        optim=config.optimizer,
        lr_scheduler_type=config.scheduler,
        warmup_steps=config.warmup_steps,
        max_grad_norm=config.max_grad_norm,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=config.inference.seed,
        data_seed=config.inference.seed,
        bf16=config.inference.dtype == "bfloat16",
        fp16=config.inference.dtype == "float16",
        use_cpu=config.inference.device == "cpu",
        report_to="none",
        push_to_hub=False,
        logging_steps=1,
        save_strategy="steps",
        save_steps=min(config.save_steps, steps),
        save_total_limit=2,
        eval_strategy="no",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        disable_tqdm=True,
    )
    if str(args.device).split(":")[0] != config.inference.device:
        raise ValueError("Trainer selected a different device from the explicit config")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=stage_train,
        eval_dataset=stage_eval,
        data_collator=ResponseCollator(pad_id, tensorize=True),
        processing_class=tokenizer,
    )
    manifest = {
        "schema_version": "compactfacts.training.v1",
        "status": "running",
        "stage": config.stage,
        "fingerprint": fingerprint,
        "gates": gates,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
        "train_hash": stable_hash([ticket.model_dump(mode="json") for ticket in train_tickets]),
        "dev_hash": stable_hash([ticket.model_dump(mode="json") for ticket in dev_tickets]),
        "split_manifest_hash": split["manifest_hash"],
        "split_manifest_file_hash": file_hash(config.split_manifest_path),
        "baseline_evidence": baseline_evidence,
        "weight_hash_seconds": weight_hash_seconds,
        "git": _git_state(),
        "environment": _environment(torch, config),
        "model_snapshot": snapshot,
        "base_model": config.inference.model_id,
        "base_revision": config.inference.revision,
        "quantization": None,
        "precision": config.inference.dtype,
        "seed": config.inference.seed,
        "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
        "effective_batch_note": "single process; final partial accumulation may be smaller",
        "trainable_parameters": trainable,
        "optimizer_steps_planned": steps,
        "best_checkpoint": None,
        "checkpoint_note": "final adapter is a candidate; no official-score checkpoint selection",
        "command": ["compactfacts", *sys.argv[1:]],
        "privacy": "no source messages, targets or generated fact values in this manifest",
    }
    manifest_path = output / "training_manifest.json"
    write_json(manifest_path, manifest)
    started = time.perf_counter()
    try:
        initial_loss = float(trainer.evaluate()["eval_loss"])
        if not math.isfinite(initial_loss):
            raise ValueError("non_finite_initial_loss")
        trainer.train()
        final_loss = float(trainer.evaluate()["eval_loss"])
        if not math.isfinite(final_loss):
            raise ValueError("non_finite_final_loss")
        model.config.use_cache = True
        sample_tickets = dev_tickets if config.stage == "full" else train_tickets[:n]
        checks, predictions = _generation_checks(model, tokenizer, torch, config, sample_tickets)
        logged_losses = [
            float(item["loss"]) for item in trainer.state.log_history if "loss" in item
        ]
        gradient_norms = [
            float(item["grad_norm"]) for item in trainer.state.log_history if "grad_norm" in item
        ]
        gradients_finite = bool(gradient_norms) and all(map(math.isfinite, gradient_norms))
        finite_updates = (
            trainer.state.global_step > 0
            and bool(logged_losses)
            and all(map(math.isfinite, logged_losses))
            and gradients_finite
        )
        loss_decreased = final_loss < initial_loss * config.minimum_loss_ratio
        exact_generation = all(check["exact_fact_set"] for check in checks)
        passed = finite_updates
        if config.stage == "tiny_overfit":
            passed = passed and loss_decreased and exact_generation
        elif config.stage == "full":
            passed = passed and all(prediction.status == "ok" for prediction in predictions)
        gates[config.stage] = {
            "passed": passed,
            "finite_updates": finite_updates,
            "gradients_finite": gradients_finite,
            "loss_log_count": len(logged_losses),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_decreased": loss_decreased,
            "generation_exact": exact_generation,
            "generation_checks": checks,
            "smoke_note": "smoke generation diagnostic; exact generation required by tiny_overfit",
        }
        adapter_path = output / "adapter"
        model.save_pretrained(str(adapter_path), safe_serialization=True)
        tokenizer.save_pretrained(str(adapter_path))
        manifest.update(
            {
                "status": "passed" if passed else "failed",
                "gates": gates,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "optimizer_steps": trainer.state.global_step,
                "epochs": trainer.state.epoch,
                "training_log": trainer.state.log_history,
                "validation_metrics": evaluate(
                    sample_tickets, predictions, official_scorer=official_scorer
                ),
                "validation_scope": "held_out_dev"
                if config.stage == "full"
                else "tiny_training_subset",
                "adapter_path": str(adapter_path),
                "adapter_bytes": sum(
                    path.stat().st_size for path in adapter_path.rglob("*") if path.is_file()
                ),
                "conclusion": "mechanics gate only; not a quality improvement claim"
                if config.stage != "full"
                else "candidate measured; compare primary metric and resources to frozen baseline",
            }
        )
    except Exception as exc:
        manifest.update(
            status="failed",
            failure_type=type(exc).__name__,
            failure_note="local training failed; source data excluded from error record",
        )
        raise
    finally:
        manifest["duration_seconds"] = time.perf_counter() - started
        manifest["cuda_peak_allocated_bytes"] = (
            torch.cuda.max_memory_allocated() if config.inference.device == "cuda" else None
        )
        manifest["cuda_peak_scope"] = (
            "process high-water includes loading and any earlier CUDA work in the same process"
        )
        write_json(manifest_path, manifest)
    return manifest
