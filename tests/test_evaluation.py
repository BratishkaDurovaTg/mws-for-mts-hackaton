"""Hand-computed scorer examples; no extractor generates these expected labels."""

from unittest.mock import Mock

import pytest

from compactfacts.evaluation import evaluate
from compactfacts.postprocessing import fact_key, normalize_value
from compactfacts.schemas import Fact, Message, Prediction, Ticket


def gold(facts, id_="t1"):
    return Ticket(
        id=id_, messages=[Message(id="m1", role="user", content="403 404 v1.2.3")], facts=facts
    )


def prediction(facts, id_="t1", **kwargs):
    return Prediction(ticket_id=id_, status="ok", facts=facts, **kwargs)


def fact(value="403", type_="error_code", **kwargs):
    return Fact(type=type_, value=value, **kwargs)


@pytest.mark.parametrize(
    "expected,actual,tp,fp,fn,f1,exact",
    [
        ([fact()], [fact()], 1, 0, 0, 1.0, 1.0),
        ([], [], 0, 0, 0, 0.0, 1.0),
        ([fact()], [], 0, 0, 1, 0.0, 0.0),
        ([], [fact()], 0, 1, 0, 0.0, 0.0),
        ([fact()], [fact("404")], 0, 1, 1, 0.0, 0.0),
        ([fact(), fact("v1.2.3", "version")], [fact(), fact("404")], 1, 1, 1, 0.5, 0.0),
    ],
    ids=["perfect", "both-empty", "miss", "spurious", "wrong-value", "partial"],
)
def test_analytic_fact_set_scores(expected, actual, tp, fp, fn, f1, exact):
    result = evaluate([gold(expected)], [prediction(actual)])
    assert {key: result["micro"][key] for key in ["tp", "fp", "fn"]} == {
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
    assert result["micro"]["f1"] == pytest.approx(f1)
    assert result["exact_fact_set_accuracy"] == exact
    assert result["metric_status"] == "provisional"
    assert result["primary"] == {"name": "provisional_micro_f1", "score": f1}


def test_micro_and_macro_include_missed_and_spurious_classes():
    # Gold: 3 facts. Predictions: 1 TP, 2 FP, 2 FN.
    # error_code F1=1/2; version F1=0; uuid F1=0 -> macro=1/6.
    tickets = [gold([fact(), fact("v1.2.3", "version")]), gold([fact("404")], "t2")]
    predictions = [
        prediction([fact(), fact("not-a-gold-uuid", "uuid")]),
        prediction([fact("500")], "t2"),
    ]
    result = evaluate(tickets, predictions)
    assert result["micro"] == {
        "tp": 1,
        "fp": 2,
        "fn": 2,
        "precision": 1 / 3,
        "recall": 1 / 3,
        "f1": 1 / 3,
    }
    assert result["per_class"]["error_code"]["f1"] == 0.5
    assert result["macro_f1"] == pytest.approx(1 / 6)
    assert result["error_buckets"]["missing_fact"] == 2
    assert result["error_buckets"]["wrong_value"] == 1
    assert result["error_buckets"]["unmatched_fact"] == 1


@pytest.mark.parametrize("code", ["invalid_json", "invalid_schema", "output_limit"])
def test_parse_failure_on_empty_gold_never_counts_as_exact_empty_success(code):
    failed = Prediction(ticket_id="t1", status="error", facts=[], error_code=code)
    result = evaluate([gold([])], [failed])
    assert result["exact_fact_set_accuracy"] == 0.0
    assert result["success_rate"] == 0.0
    assert result["parse_failure_rate"] == 1.0
    assert result["technical_failure_rate"] == 1.0
    assert result["schema_validity_rate"] == 0.0


def test_missing_prediction_is_a_failure_even_when_gold_is_empty():
    result = evaluate([gold([])], [])
    assert result["exact_fact_set_accuracy"] == 0.0
    assert result["technical_failure_rate"] == 1.0
    assert result["counts"]["missing_predictions"] == 1
    assert result["error_buckets"]["missing_prediction"] == 1


def test_backend_failure_does_not_claim_schema_was_validated():
    failed = Prediction(ticket_id="t1", status="error", facts=[], error_code="model_unavailable")
    result = evaluate([gold([])], [failed])
    assert result["success_rate"] == 0.0
    assert result["parse_failure_rate"] == 0.0
    assert result["schema_validity_rate"] == 0.0


@pytest.mark.parametrize("succeeds", [True, False])
def test_retries_are_observable_even_when_all_attempts_fail(succeeds):
    retried = (
        prediction([], attempts=3)
        if succeeds
        else Prediction(
            ticket_id="t1", status="error", facts=[], error_code="invalid_json", attempts=3
        )
    )
    result = evaluate([gold([])], [retried])
    assert result["retried_rate"] == 1.0
    assert result["repaired_rate"] == 0.0


def test_unlabelled_gold_cannot_be_scored_as_empty():
    with pytest.raises(ValueError, match="unlabelled"):
        evaluate([Ticket(id="t1", messages=[], facts=None)], [prediction([])])


def test_empty_dataset_has_no_fabricated_perfect_score():
    with pytest.raises(ValueError, match="empty dataset"):
        evaluate([], [])


@pytest.mark.parametrize("duplicate_side", ["gold", "prediction"])
def test_duplicate_ticket_ids_are_rejected(duplicate_side):
    tickets = [gold([])]
    predictions = [prediction([])]
    if duplicate_side == "gold":
        tickets.append(gold([]))
    else:
        predictions.append(prediction([]))
    with pytest.raises(ValueError, match="duplicate ticket IDs"):
        evaluate(tickets, predictions)


def test_unknown_prediction_id_is_not_silently_ignored():
    with pytest.raises(ValueError, match="unknown ticket IDs"):
        evaluate([gold([])], [prediction([], "not-in-gold")])


def test_duplicate_facts_cannot_inflate_recall_or_tp():
    result = evaluate([gold([fact(), fact("404")])], [prediction([fact()] * 5)])
    assert result["micro"]["tp"] == 1
    assert result["micro"]["fn"] == 1
    assert result["micro"]["recall"] == 0.5
    assert result["micro"]["f1"] == pytest.approx(2 / 3)
    assert result["duplicate_fact_rate"] == 4 / 5
    assert result["error_buckets"]["duplicate"] == 4


def test_duplicate_gold_mentions_are_a_set_not_five_independent_targets():
    result = evaluate([gold([fact()] * 5)], [prediction([fact()])])
    assert result["micro"]["tp"] == 1
    assert result["micro"]["recall"] == 1.0


@pytest.mark.parametrize(
    "qualifier,value",
    [("polarity", "negated"), ("temporality", "historical"), ("certainty", "hypothetical")],
)
def test_qualifiers_are_part_of_fact_identity(qualifier, value):
    result = evaluate([gold([fact()])], [prediction([fact(**{qualifier: value})])])
    assert result["micro"]["tp"] == 0
    assert result["micro"]["fp"] == result["micro"]["fn"] == 1
    assert result["error_buckets"]["wrong_qualifier"] == 1


def test_normalization_is_opt_in_and_fact_type_scoped():
    tickets = [gold([fact("CDN", "service"), fact("bucket-17", "resource_id")])]
    predictions = [prediction([fact("cdn", "service"), fact("bucket17", "resource_id")])]
    raw = evaluate(tickets, predictions)
    normalized = evaluate(tickets, predictions, normalization={"service": "casefold"})
    assert raw["matching"] == "raw_exact"
    assert raw["micro"]["tp"] == 0
    assert normalized["matching"] == "normalized_exact"
    assert normalized["micro"]["tp"] == 1
    assert normalized["micro"]["fp"] == normalized["micro"]["fn"] == 1


@pytest.mark.parametrize(
    "type_,left,right",
    [
        ("error_code", "403", "404"),
        ("version", "v1.2.3", "v1.2.4"),
        ("resource_id", "bucket-17", "bucket17"),
        ("resource_id", "CaseSensitive", "casesensitive"),
        ("ip", "203.0.113.10", "203.0.113.11"),
    ],
)
def test_identifiers_are_not_aggressively_normalized_by_default(type_, left, right):
    assert normalize_value(left, type_) == left
    assert fact_key(fact(left, type_)) != fact_key(fact(right, type_))
    assert (
        evaluate([gold([fact(left, type_)])], [prediction([fact(right, type_)])])["micro"]["f1"]
        == 0.0
    )


def test_unknown_normalization_policy_fails_explicitly():
    with pytest.raises(ValueError, match="normalization"):
        evaluate([gold([fact()])], [prediction([fact()])], {"error_code": "semantic_guess"})


def test_official_plugin_becomes_primary_and_keeps_provisional_diagnostics():
    tickets, predictions = [gold([fact()])], [prediction([fact()])]
    scorer = Mock(
        return_value={
            "name": "official-fixture-plugin",
            "score": 0.125,
            "provenance": "synthetic-test-plugin-v1; not an actual MWS scorer",
        }
    )
    result = evaluate(tickets, predictions, official_scorer=scorer)
    scorer.assert_called_once_with(tickets, predictions)
    assert result["primary"]["score"] == 0.125
    assert result["primary"]["status"] == "official_plugin"
    assert result["micro"]["f1"] == 1.0
    assert result["metric_status"] == "provisional"


@pytest.mark.parametrize(
    "result",
    [
        None,
        [],
        {},
        {"name": "x", "score": 1.0},
        {"name": 7, "score": 1.0, "provenance": "test"},
        {"name": "x", "score": "1.0", "provenance": "test"},
        {"name": "x", "score": True, "provenance": "test"},
        {"name": "x", "score": float("nan"), "provenance": "test"},
        {"name": "x", "score": float("inf"), "provenance": "test"},
        {"name": "", "score": 1.0, "provenance": "test"},
        {"name": "x", "score": 1.0, "provenance": ""},
        {"name": "  ", "score": 1.0, "provenance": "test"},
        {"name": "x", "score": 1.0, "provenance": "  "},
    ],
)
def test_invalid_official_result_fails_without_falling_back_silently(result):
    with pytest.raises(ValueError, match="official scorer"):
        evaluate([gold([])], [prediction([])], official_scorer=Mock(return_value=result))


def test_official_plugin_exception_is_not_reported_as_provisional_success():
    with pytest.raises(RuntimeError, match="plugin failed"):
        evaluate(
            [gold([])],
            [prediction([])],
            official_scorer=Mock(side_effect=RuntimeError("plugin failed")),
        )


def test_lexical_unsupported_rate_is_not_labelled_semantic_hallucination_rate():
    result = evaluate([gold([fact()])], [prediction([fact()])])
    assert result["micro"]["f1"] == 1.0
    assert result["unsupported_evidence_rate"] == 1.0
    assert result["hallucinated_fact_rate"] is None
    assert "not a semantic" in result["hallucination_note"]
