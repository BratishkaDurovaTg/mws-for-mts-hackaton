"""Strict parsing and source grounding failure modes, including dialogue roles."""

import json

import pytest

from compactfacts.inference import error_prediction, parse_generation, predict_batch
from compactfacts.inference.audit import audit_facts
from compactfacts.inference.parser import ParseFailure, parse_payload
from compactfacts.schemas import Evidence, Fact, InferenceConfig, Message, Ticket


def ticket():
    return Ticket(id="t1", messages=[Message(id="m1", role="user", content="Ошибка 403")])


def supported_fact(**updates):
    return Fact(
        type="error_code",
        value="403",
        evidence=[Evidence(message_id="m1", start=7, end=10, text="403")],
        **updates,
    )


def payload(facts):
    return json.dumps(
        {"facts": [fact.model_dump(mode="json") for fact in facts]}, ensure_ascii=False
    )


@pytest.mark.parametrize(
    "text,code",
    [
        ("", "invalid_json"),
        ("I found no facts", "invalid_json"),
        ('{"facts": []} trailing explanation', "invalid_json"),
        ('{"facts": [], "facts": []}', "invalid_json"),
        ('{"facts": NaN}', "invalid_json"),
        ('{"facts": Infinity}', "invalid_json"),
        ('{"facts": 1e9999}', "invalid_json"),
        ("[]", "invalid_schema"),
        ("null", "invalid_schema"),
        ("{}", "invalid_schema"),
        ('{"facts": null}', "invalid_schema"),
        ('{"facts": [], "extra": "ignored?"}', "invalid_schema"),
        ('{"facts": [{"type": "error_code", "value": 403}]}', "invalid_schema"),
        ('{"facts": [{"type": "error_code", "value": ""}]}', "invalid_schema"),
        (
            '{"facts": [{"type": "error_code", "value": "403", "polarity": "maybe"}]}',
            "invalid_schema",
        ),
    ],
)
def test_malformed_or_wrong_schema_output_has_explicit_error(text, code):
    with pytest.raises(ParseFailure, match=code):
        parse_payload(text)
    result = parse_generation(ticket(), text, InferenceConfig())
    assert result.status == "error"
    assert result.error_code == code
    assert result.facts == []


def test_successful_empty_json_is_distinct_from_failed_parsing():
    result = parse_generation(ticket(), '{"facts": []}', InferenceConfig())
    assert result.status == "ok"
    assert result.facts == []
    assert result.error_code is None
    assert result.repaired is False


def test_valid_fact_preserves_evidence_qualifiers_and_token_measurement():
    extracted = supported_fact(
        polarity="negated", temporality="historical", certainty="hypothetical"
    )
    result = parse_generation(
        ticket(), payload([extracted]), InferenceConfig(), generated_tokens=17
    )
    assert result.status == "ok"
    assert result.facts == [extracted]
    assert result.generated_tokens == 17


def test_fence_repair_is_explicit_opt_in_and_observable():
    fenced = '```json\n{"facts": []}\n```'
    strict = parse_generation(ticket(), fenced, InferenceConfig(parser_mode="strict"))
    repaired = parse_generation(ticket(), fenced, InferenceConfig(parser_mode="fence"))
    assert strict.status == "error"
    assert strict.error_code == "invalid_json"
    assert repaired.status == "ok"
    assert repaired.repaired is True


@pytest.mark.parametrize(
    "text",
    [
        'Explanation\n```json\n{"facts": []}\n```',
        '```json\n{"facts": []}\n```\nAnother response',
        '{"facts": []}\n{"facts": []}',
    ],
)
def test_fence_mode_does_not_guess_json_from_arbitrary_prose_or_multiple_results(text):
    with pytest.raises(ParseFailure, match="invalid_json"):
        parse_payload(text, mode="fence")


def test_unknown_parser_mode_is_a_configuration_error():
    with pytest.raises(ValueError, match="unknown parser mode"):
        parse_payload('{"facts": []}', mode="best_effort")


def test_output_limit_is_an_error_not_an_empty_prediction():
    result = parse_generation(ticket(), " " * (2 * 1024**2 + 1), InferenceConfig())
    assert result.status == "error"
    assert result.error_code == "output_limit"


def test_output_fact_type_allowlist_is_optional_but_enforced_when_given():
    text = payload([supported_fact()])
    allowed = parse_generation(ticket(), text, InferenceConfig(allowed_types=["error_code"]))
    rejected = parse_generation(ticket(), text, InferenceConfig(allowed_types=["version"]))
    assert allowed.status == "ok"
    assert rejected.status == "error"
    assert rejected.error_code == "unknown_fact_type"


def test_evidence_is_required_by_default_and_can_be_diagnostically_opted_out():
    text = payload([Fact(type="error_code", value="403")])
    strict = parse_generation(ticket(), text, InferenceConfig())
    diagnostic = parse_generation(ticket(), text, InferenceConfig(require_evidence=False))
    assert strict.status == "error"
    assert strict.error_code == "unsupported_evidence"
    assert diagnostic.status == "ok"
    assert audit_facts(ticket(), diagnostic.facts)[0]["supported"] is False


@pytest.mark.parametrize(
    "fact,issues",
    [
        (Fact(type="error_code", value="403"), {"missing_evidence"}),
        (
            Fact(
                type="error_code",
                value="403",
                evidence=[Evidence(message_id="missing", start=7, end=10, text="403")],
            ),
            {"unknown_source_message"},
        ),
        (
            Fact(
                type="error_code",
                value="403",
                evidence=[Evidence(message_id="m1", start=0, end=3, text="403")],
            ),
            {"invalid_source_span"},
        ),
        (
            Fact(
                type="error_code",
                value="404",
                evidence=[Evidence(message_id="m1", start=7, end=10, text="403")],
            ),
            {"value_not_in_evidence", "value_not_in_source"},
        ),
    ],
)
def test_lexical_audit_reports_specific_grounding_failure(fact, issues):
    finding = audit_facts(ticket(), [fact])[0]
    assert finding["supported"] is False
    assert set(finding["issues"]) == issues
    assert finding["fact_index"] == 0


def test_evidence_cannot_be_flattened_across_messages():
    dialogue = Ticket(
        id="dialogue",
        messages=[
            Message(id="u1", role="user", content="Ошибка 502"),
            Message(id="s1", role="support", content="Ошибка 403?"),
        ],
    )
    # A correct substring from the support message cannot be attributed to the user.
    false_source = Fact(
        type="error_code",
        value="403",
        evidence=[Evidence(message_id="u1", start=7, end=10, text="403")],
    )
    assert audit_facts(dialogue, [false_source])[0]["issues"] == ["invalid_source_span"]
    source_preserved = Fact(
        type="error_code",
        value="403",
        evidence=[Evidence(message_id="s1", start=7, end=10, text="403")],
    )
    assert audit_facts(dialogue, [source_preserved])[0]["supported"] is True
    message = {message.id: message for message in dialogue.messages}[
        source_preserved.evidence[0].message_id
    ]
    assert message.role == "support"
    assert message.content.endswith("?")
    # Lexical support intentionally does not turn that operator question into a user fact.


def test_lexical_support_does_not_certify_negation_or_semantic_truth():
    text = "Ошибки 403 нет"
    source = Ticket(id="negation", messages=[Message(id="m1", role="user", content=text)])
    wrong_polarity = Fact(
        type="error_code",
        value="403",
        polarity="affirmed",
        evidence=[Evidence(message_id="m1", start=0, end=len(text), text=text)],
    )
    finding = audit_facts(source, [wrong_polarity])[0]
    assert finding["supported"] is True
    assert finding["issues"] == []
    # Gold qualifier scoring, not lexical matching, must reject the wrong affirmation.


def test_error_envelope_retains_ticket_identity_and_token_cost():
    result = error_prediction("t-cost", "invalid_json", generated_tokens=23)
    assert result.ticket_id == "t-cost"
    assert result.status == "error"
    assert result.facts == []
    assert result.generated_tokens == 23


def test_predict_batch_keeps_input_order_and_individual_failures():
    class Extractor:
        def predict(self, source):
            return error_prediction(source.id, "synthetic_backend_failure")

    inputs = [Ticket(id="b", messages=[]), Ticket(id="a", messages=[])]
    outputs = predict_batch(Extractor(), inputs)
    assert [output.ticket_id for output in outputs] == ["b", "a"]
    assert all(output.status == "error" for output in outputs)
