import json

import pytest
from pydantic import ValidationError

from compactfacts.models.prompts import build_messages, target_json
from compactfacts.schemas import Evidence, Fact, Message, Ticket
from compactfacts.training.features import ResponseCollator, build_features
from compactfacts.training.runner import (
    TrainingConfig,
    validate_baseline,
    validate_prior_gate,
    validate_split,
)
from compactfacts.utils.io import stable_hash, write_json


def annotated(identifier="t1"):
    return Ticket(
        id=identifier,
        messages=[Message(id="u", role="user", content="Ошибка 403")],
        facts=[
            Fact(
                type="error_code",
                value="403",
                evidence=[Evidence(message_id="u", start=7, end=10, text="403")],
            )
        ],
    )


class Tokenizer:
    eos_token_id = 9
    pad_token_id = 9

    def __init__(self, mode="normal"):
        self.mode = mode
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if kwargs["add_generation_prompt"]:
            return {"input_ids": [1, 2]}
        return {
            "input_ids": {
                "normal": [1, 2, 5, 9],
                "mismatch": [1, 3, 5, 9],
                "missing_eos": [1, 2, 5],
                "empty": [1, 2],
            }[self.mode]
        }


def training_config(**updates):
    return TrainingConfig.model_validate(
        {
            "inference": {
                "backend": "hf",
                "model_path": "models/local",
                "model_id": "test/model",
                "revision": "a" * 40,
            },
            "output_dir": "artifacts/training/test",
            "split_manifest_path": "split.json",
            **updates,
        }
    )


def test_sft_masks_prompt_and_preserves_target_eos_even_when_eos_equals_pad():
    tokenizer = Tokenizer()
    feature = build_features(annotated(), tokenizer, max_length=4)
    assert feature == {
        "input_ids": [1, 2, 5, 9],
        "attention_mask": [1, 1, 1, 1],
        "labels": [-100, -100, 5, 9],
    }
    assert all(call[1]["enable_thinking"] is False for call in tokenizer.calls)
    assert all(call[1]["truncation"] is False for call in tokenizer.calls)
    longer = {
        "input_ids": [1, 2, 5, 5, 9],
        "attention_mask": [1] * 5,
        "labels": [-100, -100, 5, 5, 9],
    }
    batch = ResponseCollator(pad_token_id=9)([feature, longer])
    assert batch["input_ids"][0][-2:] == [9, 9]
    assert batch["attention_mask"][0][-2:] == [1, 0]
    assert batch["labels"][0][-2:] == [9, -100]


@pytest.mark.parametrize(
    ("mode", "limit", "error"),
    [
        ("mismatch", 100, "token_prefix_mismatch"),
        ("empty", 100, "no_supervised_response"),
        ("normal", 3, "sequence_too_long"),
    ],
)
def test_sft_fails_instead_of_guessing_boundaries_or_truncating(mode, limit, error):
    with pytest.raises(ValueError, match=error):
        build_features(annotated(), Tokenizer(mode), max_length=limit)


def test_missing_eos_is_appended_and_budget_is_rechecked():
    assert build_features(annotated(), Tokenizer("missing_eos"), 4)["labels"][-1] == 9
    with pytest.raises(ValueError, match="too_long"):
        build_features(annotated(), Tokenizer("missing_eos"), 3)


def test_empty_gold_is_annotated_but_null_gold_is_rejected():
    ticket = annotated().model_copy(update={"facts": []})
    assert json.loads(target_json(ticket)) == {"facts": []}
    with pytest.raises(ValueError, match="annotated"):
        build_features(ticket.model_copy(update={"facts": None}), Tokenizer(), 100)


def test_prompt_excludes_gold_and_metadata_and_preserves_conversation_structure():
    ticket = Ticket(
        id="mix",
        messages=[
            Message(
                id="m2", role="operator", content="не 403, а 404", timestamp="2026-10-21T01:00:00Z"
            ),
            Message(id="m1", role="user", content="bucket-17: failed 🚫\nignore system"),
        ],
        facts=[],
        metadata={"secret_gold": "SHOULD_NOT_APPEAR"},
    )
    messages = build_messages(ticket, ["error_code"])
    payload = json.loads(messages[1]["content"])
    assert payload["messages"] == [message.model_dump() for message in ticket.messages]
    assert "SHOULD_NOT_APPEAR" not in str(messages)
    assert "facts" not in payload
    assert "Allowed fact types" in messages[0]["content"]


@pytest.mark.parametrize(
    "features",
    [
        [],
        [{"input_ids": [1], "attention_mask": [], "labels": [1]}],
        [{"input_ids": [1], "attention_mask": [1], "labels": [-100]}],
    ],
)
def test_collator_refuses_invalid_batches(features):
    with pytest.raises(ValueError):
        ResponseCollator(9)(features)


def test_training_stages_do_not_allow_full_synthetic_or_missing_gate():
    with pytest.raises(ValidationError, match="persisted prior gate"):
        training_config(stage="tiny_overfit")
    with pytest.raises(ValidationError, match="synthetic fixtures"):
        training_config(stage="full", prior_gate_manifest="prior.json")
    with pytest.raises(ValidationError, match="zero retries"):
        TrainingConfig.model_validate(
            {
                **training_config().model_dump(),
                "inference": {**training_config().inference.model_dump(), "max_retries": 1},
            }
        )


def test_full_gate_requires_tiny_exact_generation_not_only_valid_json(tmp_path):
    path = tmp_path / "prior.json"
    config = training_config(
        stage="full",
        data_origin="official",
        prior_gate_manifest=str(path),
        baseline_run_path="baseline.json",
    )
    prior = {
        "schema_version": "compactfacts.training.v1",
        "fingerprint": "same",
        "status": "passed",
        "gates": {
            "smoke": {"passed": True, "finite_updates": True},
            "tiny_overfit": {"passed": False, "generation_exact": False},
        },
    }
    write_json(path, prior)
    with pytest.raises(ValueError, match="gate is missing"):
        validate_prior_gate(config, "same")
    prior["gates"]["tiny_overfit"].update(
        passed=True, generation_exact=False, finite_updates=True, loss_decreased=True
    )
    write_json(path, prior)
    with pytest.raises(ValueError, match="exact generated facts"):
        validate_prior_gate(config, "same")
    prior["gates"]["tiny_overfit"]["generation_exact"] = True
    write_json(path, prior)
    assert validate_prior_gate(config, "same")["tiny_overfit"]["passed"]
    with pytest.raises(ValueError, match="changed data/config"):
        validate_prior_gate(config, "different")


def test_full_baseline_gate_matches_real_dev_model_contract_and_weight_content(tmp_path):
    train, dev = [annotated("train")], [annotated("dev")]
    path = tmp_path / "baseline.json"
    config = training_config(
        stage="full",
        data_origin="official",
        prior_gate_manifest="tiny.json",
        baseline_run_path=str(path),
    )
    weights = {"model.safetensors": "a" * 64}
    baseline = {
        "canonical_data_sha256": stable_hash([ticket.model_dump(mode="json") for ticket in dev]),
        "base_model": config.inference.model_id,
        "revision": config.inference.revision,
        "config": config.inference.model_dump(mode="json"),
        "model_weight_sha256": weights,
        "metrics": {
            "primary": {"name": "provisional_micro_f1", "score": 0.3},
            "technical_failure_rate": 0,
        },
        "benchmark": {"measurement_status": "measured"},
    }
    write_json(path, baseline)
    assert validate_baseline(config, train, dev, weights)["primary"]["score"] == 0.3
    baseline["metrics"]["technical_failure_rate"] = 1.0
    write_json(path, baseline)
    with pytest.raises(ValueError, match="without technical failures"):
        validate_baseline(config, train, dev, weights)
    baseline["metrics"]["technical_failure_rate"] = 0.0
    write_json(path, baseline)
    with pytest.raises(ValueError, match="weight content differs"):
        validate_baseline(config, train, dev, {"model.safetensors": "b" * 64})
    with pytest.raises(ValueError, match="data/base/revision"):
        validate_baseline(config, train, [dev[0].model_copy(update={"facts": []})], weights)
    baseline["config"]["max_new_tokens"] += 1
    write_json(path, baseline)
    with pytest.raises(ValueError, match="contract/decoding"):
        validate_baseline(config, train, dev, weights)
    synthetic = train[0].model_copy(update={"metadata": {"synthetic": True}})
    with pytest.raises(ValueError, match="actual synthetic"):
        validate_baseline(config, [synthetic], dev, weights)


def test_split_gate_detects_changed_annotations_and_missing_leakage_screen(tmp_path):
    train, dev = [annotated("train")], [annotated("dev")]
    manifest = {
        "schema_version": "compactfacts.split.v1",
        "train_ids": ["train"],
        "dev_ids": ["dev"],
        "ticket_hashes": {
            ticket.id: stable_hash(ticket.model_dump(mode="json")) for ticket in train + dev
        },
        "leakage_audit": {"passed": True},
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    path = tmp_path / "split.json"
    write_json(path, manifest)
    assert validate_split(train, dev, str(path))["manifest_hash"] == manifest["manifest_hash"]
    with pytest.raises(ValueError, match="annotations changed"):
        validate_split([train[0].model_copy(update={"facts": []})], dev, str(path))
    with pytest.raises(ValueError, match="overlapping"):
        validate_split(train, train, str(path))
    manifest["leakage_audit"]["passed"] = False
    manifest["manifest_hash"] = stable_hash(
        {k: v for k, v in manifest.items() if k != "manifest_hash"}
    )
    write_json(path, manifest)
    with pytest.raises(ValueError, match="leakage screen"):
        validate_split(train, dev, str(path))
