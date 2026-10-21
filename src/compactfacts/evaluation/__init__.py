"""Independent provisional fact-set scoring; official scorer is replaceable."""

import math
from collections import Counter, defaultdict

from compactfacts.inference.audit import audit_facts
from compactfacts.postprocessing import fact_key
from compactfacts.schemas import Prediction, Ticket


def _scores(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate(
    gold: list[Ticket], predictions: list[Prediction], normalization=None, official_scorer=None
) -> dict:
    if not gold:
        raise ValueError("cannot evaluate an empty dataset")
    if any(t.facts is None for t in gold):
        raise ValueError("gold contains unlabelled records")
    gold_ids = [t.id for t in gold]
    pred_ids = [p.ticket_id for p in predictions]
    if len(gold_ids) != len(set(gold_ids)) or len(pred_ids) != len(set(pred_ids)):
        raise ValueError("duplicate ticket IDs")
    if set(pred_ids) - set(gold_ids):
        raise ValueError("predictions contain unknown ticket IDs")
    lookup = {p.ticket_id: p for p in predictions}
    totals = Counter()
    buckets = Counter()
    classes = defaultdict(Counter)
    for ticket in gold:
        expected = {fact_key(f, normalization) for f in ticket.facts}
        prediction = lookup.get(ticket.id)
        if prediction is not None:
            totals["retried"] += int(prediction.attempts > 1)
        failed = prediction is None or prediction.status != "ok"
        if failed:
            actual = set()
            totals["missing_predictions" if prediction is None else "technical_failures"] += 1
            code = "missing_prediction" if prediction is None else prediction.error_code
            buckets[code] += 1
        else:
            actual = {fact_key(f, normalization) for f in prediction.facts}
            totals["ok"] += 1
            totals["repaired"] += int(prediction.repaired)
            totals["predicted_mentions"] += len(prediction.facts)
            duplicates = len(prediction.facts) - len(actual)
            totals["duplicates"] += duplicates
            buckets["duplicate"] += duplicates
            for finding in audit_facts(ticket, prediction.facts):
                totals["audited_facts"] += 1
                if not finding["supported"]:
                    totals["unsupported_facts"] += 1
                    buckets["unsupported_evidence"] += 1
        matched = expected & actual
        extra, missed = actual - expected, expected - actual
        totals.update(
            tp=len(matched),
            fp=len(extra),
            fn=len(missed),
            gold=len(expected),
            predicted=len(actual),
        )
        if expected == actual and not failed:
            totals["exact"] += 1
        buckets["missing_fact"] += len(missed)
        for key in extra:
            if any(key[0] == other[0] and key[1] == other[1] for other in missed):
                buckets["wrong_qualifier"] += 1
            elif any(key[0] == other[0] for other in missed):
                buckets["wrong_value"] += 1
            elif any(key[1:] == other[1:] for other in missed):
                buckets["wrong_type"] += 1
            else:
                buckets["unmatched_fact"] += 1
        for key in expected | actual:
            classes[key[0]].update(
                tp=int(key in matched), fp=int(key in extra), fn=int(key in missed)
            )
    n = len(gold)
    per_class = {
        key: _scores(count["tp"], count["fp"], count["fn"])
        for key, count in sorted(classes.items())
    }
    parse_codes = {"invalid_json", "invalid_schema", "output_limit"}
    parse_failures = sum(buckets[x] for x in parse_codes)
    schema_valid = sum(
        p.status == "ok" or p.error_code in {"unsupported_evidence", "unknown_fact_type"}
        for p in predictions
    )
    result = {
        "metric_status": "provisional",
        "matching": "normalized_exact" if normalization else "raw_exact",
        "fact_identity": ["type", "value", "polarity", "temporality", "certainty"],
        "micro": _scores(totals["tp"], totals["fp"], totals["fn"]),
        "per_class": per_class,
        "macro_f1": sum(x["f1"] for x in per_class.values()) / len(per_class) if per_class else 0.0,
        "exact_fact_set_accuracy": totals["exact"] / n,
        "unmatched_prediction_rate": totals["fp"] / totals["predicted"]
        if totals["predicted"]
        else 0.0,
        "missed_fact_rate": totals["fn"] / totals["gold"] if totals["gold"] else 0.0,
        "schema_validity_rate": schema_valid / n,
        "parse_failure_rate": parse_failures / n,
        "technical_failure_rate": (totals["technical_failures"] + totals["missing_predictions"])
        / n,
        "success_rate": totals["ok"] / n,
        "repaired_rate": totals["repaired"] / n,
        "retried_rate": totals["retried"] / n,
        "duplicate_fact_rate": totals["duplicates"] / totals["predicted_mentions"]
        if totals["predicted_mentions"]
        else 0.0,
        "unsupported_evidence_rate": totals["unsupported_facts"] / totals["audited_facts"]
        if totals["audited_facts"]
        else 0.0,
        "hallucinated_fact_rate": None,
        "hallucination_note": (
            "Lexical grounding is diagnostic, not a semantic hallucination verdict."
        ),
        "counts": dict(totals),
        "records": n,
        "error_buckets": dict(sorted(buckets.items())),
        "zero_division": 0,
        "empty_set_policy": "both-empty successful record is exact; empty micro F1=0",
    }
    result["primary"] = {"name": "provisional_micro_f1", "score": result["micro"]["f1"]}
    if official_scorer is not None:
        official = official_scorer(gold, predictions)
        if (
            not isinstance(official, dict)
            or not isinstance(official.get("name"), str)
            or not isinstance(official.get("provenance"), str)
            or not official["name"].strip()
            or not official["provenance"].strip()
            or type(official.get("score")) not in {int, float}
            or not math.isfinite(official["score"])
        ):
            raise ValueError("official scorer requires finite score, name and provenance")
        result["primary"] = {**official, "status": "official_plugin"}
    return result
