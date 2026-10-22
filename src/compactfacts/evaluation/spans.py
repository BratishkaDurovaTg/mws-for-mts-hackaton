"""Occurrence-aware exact span diagnostic, separate from MWS/provisional fact-set F1."""

from compactfacts.inference.audit import audit_facts
from compactfacts.schemas import Prediction, Ticket


def _span_set(ticket, facts):
    if any(not x["supported"] for x in audit_facts(ticket, facts)):
        raise ValueError("span evaluation requires valid literal evidence")
    if any(ev.text != fact.value for fact in facts for ev in fact.evidence):
        raise ValueError("span diagnostic requires evidence to be the exact extracted value")
    return {(fact.type, ev.message_id, ev.start, ev.end) for fact in facts for ev in fact.evidence}


def evaluate_spans(gold: list[Ticket], predictions: list[Prediction]) -> dict:
    if not gold or any(t.facts is None for t in gold):
        raise ValueError("span diagnostic requires nonempty labelled gold")
    ids = [t.id for t in gold]
    pred_ids = [p.ticket_id for p in predictions]
    if len(ids) != len(set(ids)) or len(pred_ids) != len(set(pred_ids)):
        raise ValueError("duplicate ticket IDs")
    if set(pred_ids) - set(ids):
        raise ValueError("unknown prediction ticket IDs")
    lookup = {p.ticket_id: p for p in predictions}
    tp = fp = fn = invalid = 0
    for ticket in gold:
        expected = _span_set(ticket, ticket.facts)
        pred = lookup.get(ticket.id)
        if pred is None or pred.status != "ok":
            actual, invalid = set(), invalid + 1
        else:
            try:
                actual = _span_set(ticket, pred.facts)
            except ValueError:
                actual, invalid = set(), invalid + 1
        tp += len(expected & actual)
        fp += len(actual - expected)
        fn += len(expected - actual)
    return {
        "metric_status": "public_auxiliary_diagnostic_not_mws",
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "failed_records": invalid,
        "failure_rate": invalid / len(gold),
        "identity": ["type", "message_id", "start", "end"],
        "scope": "exact span diagnostic only; ignores qualifiers/relations/normalized values",
    }
