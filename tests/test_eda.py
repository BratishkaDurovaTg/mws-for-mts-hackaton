import json

import pytest

from compactfacts.data import analyze
from compactfacts.schemas import Fact, Message, Ticket


def ticket(identifier, text, facts=None, role="user", metadata=None):
    return Ticket(
        id=identifier,
        messages=[Message(id="m", role=role, content=text)],
        facts=facts,
        metadata=metadata or {},
    )


def test_aggregate_scope_and_null_empty_labels_are_distinct():
    items = [
        ticket("a", "Привет 😀", None),
        ticket("b", "hello there", []),
        ticket("c", "Ошибка API", [Fact(type="error", value="403")]),
    ]
    report = analyze(items)
    assert report["scope"]["records_inspected"] == 3
    assert report["scope"]["all_supplied_records_inspected"] is True
    assert report["counts"]["labelled_tickets"] == 2
    assert report["counts"]["unlabelled_tickets"] == 1
    assert report["counts"]["labelled_empty_tickets"] == 1
    assert report["counts"]["facts"] == 1
    assert report["schema"]["null_counts"]["facts"] == 1
    assert report["lengths"]["ticket_characters"]["min"] == len("Привет 😀")
    assert report["lengths"]["ticket_model_tokens"] is None
    assert report["language_heuristic"]["tickets"] == {"ru": 1, "en": 1, "mixed": 1, "unknown": 0}


def test_real_tokenizer_hook_is_separate_from_whitespace_lengths():
    # A deterministic token-ID fixture verifies the hook, not linguistic quality.
    report = analyze(
        [ticket("a", "two words")], tokenizer=lambda text: {"input_ids": [1, 2, 3, 4, 5]}
    )
    assert report["lengths"]["ticket_whitespace_units"]["mean"] == 2
    assert report["lengths"]["ticket_model_tokens"]["mean"] == 5
    assert report["lengths"]["message_model_tokens"]["max"] == 5
    assert report["lengths"]["unavailable_token_reason"] is None
    assert (
        analyze([ticket("a", "x")], tokenizer=lambda text: 7)["lengths"]["ticket_model_tokens"][
            "max"
        ]
        == 7
    )


@pytest.mark.parametrize(
    "value", ["fake tokens", None, True, -1, {"input_ids": [[1, 2]]}, ["one", "two"]]
)
def test_invalid_tokenizer_output_fails_instead_of_fabricating_counts(value):
    with pytest.raises(ValueError, match="token"):
        analyze([ticket("a", "x")], tokenizer=lambda text: value)


def test_sensitive_text_ids_labels_roles_and_metadata_keys_do_not_leak():
    item = ticket(
        "SECRET-TICKET-ID",
        "Secret raw ticket support body",
        [Fact(type="SECRET-LABEL", value="SECRET-VALUE")],
        role="SECRET-ROLE",
        metadata={"SECRET-METADATA-KEY": "SECRET-METADATA-VALUE"},
    )
    report = analyze([item])
    serialized = json.dumps(report)
    for sensitive in (
        "SECRET-TICKET-ID",
        "Secret raw ticket support body",
        "SECRET-LABEL",
        "SECRET-VALUE",
        "SECRET-ROLE",
        "SECRET-METADATA-KEY",
        "SECRET-METADATA-VALUE",
    ):
        assert sensitive not in serialized
    assert report["privacy"]["pseudonyms_are_anonymization"] is False
    assert sum(report["fact_types"].values()) == 1
    assert sum(role["messages"] for role in report["roles"].values()) == 1


def test_conflicting_annotations_and_mixed_coverage_are_reported_without_deletion():
    items = [
        ticket("a", "Identical text", [Fact(type="error", value="403")]),
        ticket("b", "Identical text", [Fact(type="error", value="404")]),
        ticket("c", "Identical text", None),
    ]
    report = analyze(items)
    duplicates = report["duplicates"]
    assert duplicates["exact_text_groups"] == 1
    assert duplicates["exact_text_excess_records"] == 2
    assert duplicates["conflicting_annotation_groups"] == 1
    assert duplicates["mixed_label_coverage_exact_groups"] == 1
    assert duplicates["automatic_deletion"] is False
    assert len(items) == report["counts"]["tickets"] == 3


def test_all_requested_pattern_diagnostics_have_observed_counts():
    text = (
        "10.0.0.1 invalid 999.999.999.999 ERR_TIMEOUT 403 v1.2.3 "
        "550e8400-e29b-41d4-a716-446655440000 https://example.invalid "
        "2026-10-21T12:00:00Z\nTraceback (most recent call last)\n```python\ndef run():\nраypal"
    )
    report = analyze([ticket("a", text)])
    assert report["patterns"]["ipv4"]["occurrences"] == 1
    for pattern in (
        "uuid",
        "error_code",
        "version",
        "url",
        "stacktrace",
        "code",
        "timestamp",
        "mixed_script_word",
    ):
        assert report["patterns"][pattern]["tickets"] == 1
        assert report["patterns"][pattern]["messages"] == 1


def test_distribution_reports_robust_and_classical_outlier_sensitivity():
    items = [ticket(str(i), "x" * size) for i, size in enumerate([1, 1, 1, 100])]
    report = analyze(items)
    summary = report["lengths"]["ticket_characters"]
    assert summary["mean"] == 25.75
    assert summary["median"] == 1
    assert summary["mad"] == 0
    assert summary["iqr_outlier_count"] == 1
    assert summary["mean_without_iqr_outliers"] == 1
    assert report["counts"]["tickets"] == 4


def test_empty_scope_is_valid_and_does_not_emit_nan():
    report = analyze([])
    assert report["counts"]["tickets"] == 0
    assert report["lengths"]["ticket_characters"]["mean"] is None
    json.dumps(report, allow_nan=False)


def test_oversized_analysis_refuses_instead_of_silent_sampling(monkeypatch):
    monkeypatch.setattr("compactfacts.data.split.MAX_TICKETS", 1)
    with pytest.raises(ValueError, match="budget"):
        analyze([ticket("a", "a"), ticket("b", "b")])
