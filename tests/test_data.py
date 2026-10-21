import json

import pytest

from compactfacts.data import (
    CanonicalAdapter,
    MappingAdapter,
    export_submission,
    read_tickets,
    write_tickets,
)
from compactfacts.schemas import Fact, Message, Prediction, Ticket


def ticket(identifier="case-a", facts=None):
    return Ticket(
        id=identifier,
        messages=[Message(id="m0", role="user", content="Ошибка 403\nIP 10.0.0.1 😀")],
        facts=facts,
        metadata={"source": "synthetic"},
    )


def test_roundtrip_preserves_unicode_roles_and_label_absence(tmp_path):
    items = [
        ticket("one", None),
        ticket("two", []),
        ticket("three", [Fact(type="error", value="403")]),
    ]
    path = tmp_path / "tickets.jsonl"
    write_tickets(path, items)
    restored = read_tickets(path)
    assert restored == items
    assert restored[0].facts is None
    assert restored[1].facts == []
    assert restored[0].messages[0].content == "Ошибка 403\nIP 10.0.0.1 😀"
    assert path.stat().st_mode & 0o777 == 0o600


def test_canonical_adapter_is_strict():
    record = ticket().model_dump(mode="json")
    record["unknown"] = "field"
    with pytest.raises(ValueError):
        CanonicalAdapter().to_ticket(record)


def test_two_explicit_mappings_produce_same_canonical_ticket():
    plain = MappingAdapter({"id": "case", "text": "body", "facts": "annotations"})
    dialog = MappingAdapter(
        {"id": "record", "messages": "conversation", "facts": "labels"},
        message_mapping={"id": "key", "role": "speaker", "content": "utterance"},
        fact_mapping={"type": "slot", "value": "raw_value"},
    )
    a = plain.to_ticket(
        {"case": "42", "body": "error 403", "annotations": [{"type": "error", "value": "403"}]}
    )
    b = dialog.to_ticket(
        {
            "record": "42",
            "conversation": [{"key": "m0", "speaker": "user", "utterance": "error 403"}],
            "labels": [{"slot": "error", "raw_value": "403"}],
        }
    )
    assert a == b
    assert a.facts[0].value == "403"


@pytest.mark.parametrize("raw_facts", [None, []])
def test_mapping_preserves_null_versus_empty(raw_facts):
    adapter = MappingAdapter({"id": "case", "text": "body", "facts": "labels"})
    result = adapter.to_ticket({"case": "a", "body": "nothing", "labels": raw_facts})
    assert result.facts == raw_facts


@pytest.mark.parametrize(
    "mapping",
    [
        {"text": "body"},
        {"id": "i"},
        {"id": "i", "text": "body", "messages": "dialog"},
        {"id": "i", "text": "i"},
        {"id": "i", "text": ""},
        {"id": "i", "text": "body", "unsupported": "x"},
    ],
)
def test_mapping_config_rejects_ambiguous_or_unknown_fields(mapping):
    with pytest.raises(ValueError):
        MappingAdapter(mapping)


@pytest.mark.parametrize(
    "record",
    [
        {"body": "text"},
        {"case": "i"},
        {"case": 12, "body": "text"},
        {"case": "i", "body": "text", "labels": "not-array"},
    ],
)
def test_mapping_does_not_infer_or_coerce_bad_records(record):
    with pytest.raises(ValueError):
        MappingAdapter({"id": "case", "text": "body", "facts": "labels"}).to_ticket(record)


def test_reader_uses_explicit_adapter(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps({"case": "a", "body": "plain"}) + "\n")
    adapter = MappingAdapter({"id": "case", "text": "body"})
    assert read_tickets(path, adapter)[0].messages[0].content == "plain"
    with pytest.raises(ValueError, match="record 1"):
        read_tickets(path)


@pytest.mark.parametrize(
    "payload",
    [
        '{"id":"a","id":"b"}\n',
        '{"id":NaN}\n',
        '{"id":Infinity}\n',
        '{"id":1e400}\n',
        "[1,2]\n",
        "\n",
        "{broken}\n",
    ],
)
def test_reader_rejects_malformed_json_without_skipping(tmp_path, payload):
    path = tmp_path / "bad.jsonl"
    path.write_text(payload)
    with pytest.raises(ValueError):
        read_tickets(path)


def test_reader_invalid_record_errors_do_not_echo_source(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"secret": "sensitive-value"}) + "\n")
    with pytest.raises(ValueError) as exc:
        read_tickets(path)
    assert "sensitive-value" not in str(exc.value)


def test_duplicate_ids_and_input_budgets_fail_visibly(tmp_path):
    path = tmp_path / "tickets.jsonl"
    write_tickets(path, [ticket("a"), ticket("b")])
    with pytest.raises(ValueError, match="budget"):
        read_tickets(path, max_bytes=2)
    with pytest.raises(ValueError, match="budget"):
        read_tickets(path, max_records=1)
    path.write_text((json.dumps(ticket().model_dump(mode="json")) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        read_tickets(path)
    with pytest.raises(ValueError, match="duplicate"):
        write_tickets(tmp_path / "duplicate.jsonl", [ticket(), ticket()])


def test_official_export_fails_closed_and_canonical_preserves_errors(tmp_path):
    predictions = [
        Prediction(ticket_id="a", status="error", facts=[], error_code="offline_model_missing")
    ]
    official_path = tmp_path / "official.jsonl"
    with pytest.raises(NotImplementedError, match="official"):
        export_submission(official_path, predictions, official=True)
    assert not official_path.exists()
    diagnostic = tmp_path / "diagnostic.jsonl"
    export_submission(diagnostic, predictions)
    record = json.loads(diagnostic.read_text())
    assert record["status"] == "error"
    assert record["error_code"] == "offline_model_missing"


def test_export_can_use_an_explicit_official_adapter(tmp_path):
    class ExplicitAdapter:
        def convert(self, predictions):
            return [{"case": p.ticket_id, "slots": []} for p in predictions]

    path = tmp_path / "official.jsonl"
    export_submission(
        path,
        [Prediction(ticket_id="a", status="ok", facts=[])],
        official=True,
        adapter=ExplicitAdapter(),
    )
    assert json.loads(path.read_text()) == {"case": "a", "slots": []}
