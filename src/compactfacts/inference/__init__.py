"""Parse, validate, and distinguish technical failures from empty results."""

from compactfacts.inference.audit import audit_facts
from compactfacts.inference.parser import ParseFailure, parse_payload
from compactfacts.schemas import InferenceConfig, Prediction, Ticket


def error_prediction(ticket_id: str, code: str, generated_tokens=None) -> Prediction:
    return Prediction(
        ticket_id=ticket_id,
        status="error",
        facts=[],
        error_code=code,
        generated_tokens=generated_tokens,
    )


def parse_generation(
    ticket: Ticket, text: str, config: InferenceConfig, generated_tokens=None
) -> Prediction:
    try:
        payload, repaired = parse_payload(text, config.parser_mode)
    except ParseFailure as exc:
        return error_prediction(ticket.id, str(exc), generated_tokens)
    if config.allowed_types is not None:
        if any(f.type not in config.allowed_types for f in payload.facts):
            return error_prediction(ticket.id, "unknown_fact_type", generated_tokens)
    if config.require_evidence and any(
        not x["supported"] for x in audit_facts(ticket, payload.facts)
    ):
        return error_prediction(ticket.id, "unsupported_evidence", generated_tokens)
    return Prediction(
        ticket_id=ticket.id,
        status="ok",
        facts=payload.facts,
        repaired=repaired,
        generated_tokens=generated_tokens,
    )


def predict_batch(extractor, tickets: list[Ticket]) -> list[Prediction]:
    return [extractor.predict(ticket) for ticket in tickets]
