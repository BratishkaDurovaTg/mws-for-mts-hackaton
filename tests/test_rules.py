import pytest

from compactfacts.inference.audit import audit_facts
from compactfacts.models import create_extractor
from compactfacts.schemas import InferenceConfig, Message, Ticket


@pytest.mark.parametrize(
    "text,kind,value,polarity,temporality,certainty",
    [
        ("HTTP 403", "error_code", "403", "affirmed", "unknown", "asserted"),
        ("ошибки 403 нет", "error_code", "403", "negated", "unknown", "asserted"),
        (
            "Раньше ошибка 404, сейчас работает.",
            "error_code",
            "404",
            "affirmed",
            "historical",
            "asserted",
        ),
        ("Maybe error 403", "error_code", "403", "affirmed", "unknown", "hypothetical"),
        ("version v1.2.3", "version", "v1.2.3", "affirmed", "unknown", "asserted"),
        ("Адрес 192.168.1.17", "ip", "192.168.1.17", "affirmed", "unknown", "asserted"),
        (
            "Открыть https://example.org/a.",
            "url",
            "https://example.org/a",
            "affirmed",
            "unknown",
            "asserted",
        ),
        ("Ресурс bucket-17", "resource_id", "bucket-17", "affirmed", "unknown", "asserted"),
    ],
)
def test_exact_mention_grounding_and_qualifiers(
    text, kind, value, polarity, temporality, certainty
):
    ticket = Ticket(id="t", messages=[Message(id="m", role="user", content=text)])
    pred = create_extractor(InferenceConfig()).predict(ticket)
    found = next(f for f in pred.facts if f.type == kind and f.value == value)
    assert (found.polarity, found.temporality, found.certainty) == (
        polarity,
        temporality,
        certainty,
    )
    assert all(x["supported"] for x in audit_facts(ticket, pred.facts))


def test_invalid_ip_and_date_and_ip_not_version():
    ticket = Ticket(
        id="t", messages=[Message(id="m", role="user", content="999.1.2.3 2026-02-30 1.2.3.4")]
    )
    facts = create_extractor(InferenceConfig()).predict(ticket).facts
    assert [(f.type, f.value) for f in facts] == [("ip", "1.2.3.4")]


def test_repeated_mentions_preserve_source_messages_without_duplicate_fact():
    ticket = Ticket(
        id="t",
        messages=[
            Message(id="a", role="user", content="HTTP 403"),
            Message(id="b", role="agent", content="HTTP 403"),
        ],
    )
    pred = create_extractor(InferenceConfig()).predict(ticket)
    assert len(pred.facts) == 1
    assert [e.message_id for e in pred.facts[0].evidence] == ["a", "b"]


def test_long_input_is_explicit_error():
    pred = create_extractor(InferenceConfig(max_input_chars=3)).predict(
        Ticket(id="t", messages=[Message(id="m", role="user", content="HTTP 403")])
    )
    assert pred.status == "error" and pred.error_code == "input_too_long"


def test_repeatability_and_explicit_type_allowlist():
    engine = create_extractor(InferenceConfig(allowed_types=["error_code"]))
    ticket = Ticket(id="t", messages=[Message(id="m", role="user", content="HTTP 403 v1.2.3")])
    assert engine.predict(ticket) == engine.predict(ticket)
    assert [f.type for f in engine.predict(ticket).facts] == ["error_code"]
