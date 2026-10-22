import pytest

from compactfacts.data.spans import fact_from_span
from compactfacts.evaluation import evaluate
from compactfacts.evaluation.spans import evaluate_spans
from compactfacts.schemas import Message, Prediction, Ticket


def test_copy_preserves_unicode_ids_punctuation_nested_occurrences():
    ticket = Ticket(
        id="t", messages=[Message(id="m", role="user", content="🙂 bucket-17 / bucket-17")]
    )
    a = fact_from_span(ticket, "m", 2, 11, "resource")
    b = fact_from_span(ticket, "m", 14, 23, "resource")
    nested = fact_from_span(ticket, "m", 9, 11, "number")
    assert a.value == b.value == "bucket-17"
    assert nested.value == "17"
    ticket.facts = [a, b, nested]
    prediction = Prediction(ticket_id="t", status="ok", facts=[a, nested])
    assert evaluate([ticket], [prediction])["micro"]["fn"] == 0
    diagnostic = evaluate_spans([ticket], [prediction])
    assert (diagnostic["tp"], diagnostic["fn"]) == (2, 1)


@pytest.mark.parametrize(
    "message,start,end",
    [("missing", 0, 1), ("m", -1, 1), ("m", 0, 0), ("m", 0, 999), ("m", True, 2)],
)
def test_bad_spans_fail_before_slicing(message, start, end):
    ticket = Ticket(id="t", messages=[Message(id="m", role="user", content="abc")])
    with pytest.raises(ValueError):
        fact_from_span(ticket, message, start, end, "slot")


def test_context_evidence_is_not_misrepresented_as_a_precise_entity_span():
    from compactfacts.schemas import Evidence, Fact

    ticket = Ticket(id="t", messages=[Message(id="m", role="user", content="bucket-17")])
    ticket.facts = [fact_from_span(ticket, "m", 7, 9, "number")]
    prediction = Prediction(
        ticket_id="t",
        status="ok",
        facts=[
            Fact(
                type="number",
                value="17",
                evidence=[Evidence(message_id="m", start=0, end=9, text="bucket-17")],
            )
        ],
    )
    diagnostic = evaluate_spans([ticket], [prediction])
    assert diagnostic["failed_records"] == 1 and diagnostic["tp"] == 0
