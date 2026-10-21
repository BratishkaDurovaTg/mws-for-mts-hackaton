"""Provisional representation contracts, independent of any MWS field names."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from compactfacts.data.adapters import read_tickets
from compactfacts.inference.audit import audit_facts
from compactfacts.schemas import Evidence, Fact, InferenceConfig, Message, Prediction, Ticket

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic.jsonl"


@pytest.mark.parametrize("field,value", [("type", ""), ("value", ""), ("value", 403)])
def test_fact_requires_nonempty_strict_strings(field, value):
    data = {"type": "error_code", "value": "403", field: value}
    with pytest.raises(ValidationError):
        Fact.model_validate(data)


def test_open_fact_types_allow_future_adapter_without_schema_rewrite():
    fact = Fact(type="future_domain_specific_type", value="example")
    assert fact.type == "future_domain_specific_type"
    assert fact.evidence == []


def test_unknown_fields_fail_instead_of_being_silently_dropped():
    with pytest.raises(ValidationError):
        Fact.model_validate({"type": "error_code", "value": "403", "confidence": 1.0})


@pytest.mark.parametrize(
    "start,end,text",
    [(-1, 2, "403"), (2, 2, ""), (3, 2, "403"), (0, 2, "403"), (0, 3, "")],
)
def test_invalid_evidence_ranges_are_rejected(start, end, text):
    with pytest.raises(ValidationError):
        Evidence(message_id="m1", start=start, end=end, text=text)


def test_evidence_uses_unicode_codepoints_not_utf8_bytes():
    content = "📦 Ошибка 403"
    start = len("📦 Ошибка ")
    evidence = Evidence(message_id="m1", start=start, end=start + 3, text="403")
    ticket = Ticket(id="unicode", messages=[Message(id="m1", role="user", content=content)])
    fact = Fact(type="error_code", value="403", evidence=[evidence])
    assert start != len("📦 Ошибка ".encode())
    assert audit_facts(ticket, [fact])[0]["supported"] is True


def test_duplicate_message_ids_are_rejected():
    message = Message(id="same", role="user", content="hello")
    with pytest.raises(ValidationError, match="duplicate message id"):
        Ticket(id="t1", messages=[message, message])


def test_unlabelled_and_explicit_empty_gold_remain_distinct():
    unlabeled = Ticket(id="u", messages=[])
    empty = Ticket(id="e", messages=[], facts=[])
    assert unlabeled.model_dump()["facts"] is None
    assert empty.model_dump()["facts"] == []
    assert Ticket.model_validate(unlabeled.model_dump()).facts is None


@pytest.mark.parametrize(
    "status,facts,error_code",
    [
        ("ok", [], "invalid_json"),
        ("error", [], None),
        ("error", [Fact(type="error_code", value="403")], "invalid_json"),
    ],
)
def test_prediction_cannot_hide_failure_as_a_valid_empty_result(status, facts, error_code):
    with pytest.raises(ValidationError):
        Prediction(ticket_id="t1", status=status, facts=facts, error_code=error_code)


def test_successful_empty_and_technical_failure_are_different_contracts():
    empty = Prediction(ticket_id="t1", status="ok", facts=[])
    failure = Prediction(ticket_id="t1", status="error", facts=[], error_code="invalid_json")
    assert empty.status != failure.status
    assert failure.error_code == "invalid_json"


@pytest.mark.parametrize("attempts", [0, 4, "2"])
def test_attempts_contract_preserves_the_bounded_retry_budget(attempts):
    with pytest.raises(ValidationError):
        Prediction(ticket_id="t1", status="ok", facts=[], attempts=attempts)


@pytest.mark.parametrize(
    "updates",
    [
        {},
        {"model_path": "models/local", "model_id": "org/model"},
        {"model_path": "models/local", "model_id": "org/model", "revision": "main"},
        {
            "model_path": "models/local",
            "model_id": "org/model",
            "revision": "a" * 40,
            "enable_thinking": True,
        },
    ],
)
def test_hf_configuration_requires_local_immutable_provenance(updates):
    with pytest.raises(ValidationError):
        InferenceConfig(backend="hf", **updates)


def test_pinned_hf_config_has_bounded_deterministic_generation_settings():
    config = InferenceConfig(
        backend="hf", model_path="models/local", model_id="org/model", revision="a" * 40
    )
    assert config.enable_thinking is False
    assert config.max_retries == 0
    assert config.max_input_tokens > 0
    assert config.max_new_tokens > 0
    assert config.seed == 42


def test_synthetic_fixture_is_explicit_labelled_diverse_and_grounded():
    tickets = read_tickets(FIXTURE)
    assert len(tickets) == 20
    assert len({ticket.id for ticket in tickets}) == 20
    assert len({ticket.group_id for ticket in tickets}) >= 8
    assert all(ticket.metadata["synthetic"] is True for ticket in tickets)
    assert all(ticket.metadata["slices"] for ticket in tickets)
    assert all(ticket.facts is not None for ticket in tickets)
    for ticket in tickets:
        assert all(finding["supported"] for finding in audit_facts(ticket, ticket.facts))
    slices = {slice_ for ticket in tickets for slice_ in ticket.metadata["slices"]}
    assert {
        "empty",
        "no_facts",
        "negation",
        "correction",
        "historical",
        "current",
        "mixed_ru_en",
        "unicode",
        "long_input",
        "speaker_separation",
        "attempted_action",
        "wrong_assumption",
        "uuid",
        "ip",
        "date",
        "url",
        "version",
        "typo",
    } <= slices


def test_synthetic_labels_include_semantics_beyond_identifier_rules():
    tickets = {ticket.id: ticket for ticket in read_tickets(FIXTURE)}
    correction = tickets["syn-correction"].facts
    assert {(fact.value, fact.polarity) for fact in correction} == {
        ("403", "negated"),
        ("404", "affirmed"),
    }
    history = tickets["syn-history"].facts
    assert {(fact.temporality, fact.polarity) for fact in history} == {
        ("historical", "affirmed"),
        ("current", "negated"),
    }
    assert tickets["syn-assumption"].facts[0].certainty == "hypothetical"
    assert {fact.type for fact in tickets["syn-action"].facts} == {"action", "action_result"}
    assert max(len(message.content) for message in tickets["syn-long"].messages) > 5000


def test_intentional_duplicate_family_is_declared_without_overwriting_text():
    tickets = {ticket.id: ticket for ticket in read_tickets(FIXTURE)}
    original, copy, near = (tickets[key] for key in ["syn-http", "syn-http-copy", "syn-http-near"])
    assert original.group_id == copy.group_id == near.group_id
    assert original.messages[0].content == copy.messages[0].content
    assert original.messages[0].content != near.messages[0].content
