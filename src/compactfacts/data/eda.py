"""Aggregate-only exploratory diagnostics over an explicit, bounded ticket scope."""

from __future__ import annotations

import ipaddress
import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from compactfacts.data.split import (
    MAX_TEXT_CHARS,
    MAX_TICKETS,
    near_duplicate_pairs,
    template_fingerprint,
    text_fingerprint,
    validate_scope,
)
from compactfacts.schemas import Ticket
from compactfacts.utils.io import stable_hash

_PATTERNS = {
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "uuid": re.compile(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", re.I),
    "error_code": re.compile(r"\b(?:ERR(?:OR)?_[A-Z0-9_]+|E\d{3,}|[45]\d{2})\b", re.I),
    "version": re.compile(r"\b[vV]?\d+(?:\.\d+){1,3}(?:[-+][\w.-]+)?\b"),
    "url": re.compile(r"https?://[^\s<>]+", re.I),
    "stacktrace": re.compile(
        r"Traceback \(most recent call last\)|\bat\s+[\w.$]+\([^\n]*:\d+\)|\b\w+Exception:", re.I
    ),
    "code": re.compile(r"```|\b(?:def|class|function)\s+\w+|\b(?:SELECT|INSERT|UPDATE)\s+", re.I),
    "timestamp": re.compile(
        r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"
    ),
}
_CYRILLIC = re.compile(r"[\u0400-\u04ff]")
_LATIN = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _token(value: str) -> str:
    return "sha256:" + stable_hash(value)[:20]


def _quantile(values: list[int | float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return float(values[lower] + (values[upper] - values[lower]) * (position - lower))


def _summary(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {
            key: (0 if key == "count" else None)
            for key in (
                "count",
                "min",
                "max",
                "mean",
                "sd",
                "median",
                "q25",
                "q75",
                "iqr",
                "mad",
                "p95",
                "iqr_outlier_count",
                "mean_without_iqr_outliers",
            )
        }
    values = sorted(values)
    q25, q75 = _quantile(values, 0.25), _quantile(values, 0.75)
    iqr = q75 - q25
    median = statistics.median(values)
    retained = [v for v in values if q25 - 1.5 * iqr <= v <= q75 + 1.5 * iqr]
    return {
        "count": len(values),
        "min": values[0],
        "max": values[-1],
        "mean": statistics.fmean(values),
        "sd": statistics.pstdev(values),
        "median": median,
        "q25": q25,
        "q75": q75,
        "iqr": iqr,
        "mad": statistics.median(abs(v - median) for v in values),
        "p95": _quantile(values, 0.95),
        "iqr_outlier_count": len(values) - len(retained),
        "mean_without_iqr_outliers": statistics.fmean(retained) if retained else None,
    }


def _language(text: str) -> str:
    cyrillic, latin = bool(_CYRILLIC.search(text)), bool(_LATIN.search(text))
    return "mixed" if cyrillic and latin else "ru" if cyrillic else "en" if latin else "unknown"


def _real_token_count(tokenizer: Callable, text: str) -> int:
    encoded = tokenizer(text)
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if isinstance(encoded, int) and not isinstance(encoded, bool):
        count = encoded
    elif isinstance(encoded, Sequence) and not isinstance(encoded, (str, bytes)):
        if any(not isinstance(i, int) or isinstance(i, bool) for i in encoded):
            raise ValueError("tokenizer must return token ids or a non-negative token count")
        count = len(encoded)
    else:
        raise ValueError("tokenizer must return token ids or a non-negative token count")
    if count < 0:
        raise ValueError("tokenizer returned a negative token count")
    return count


def _fact_signature(ticket: Ticket) -> str | None:
    if ticket.facts is None:
        return None
    facts = [f.model_dump(mode="json", exclude={"evidence"}) for f in ticket.facts]
    # Annotation list ordering does not make a conflict; duplicates remain visible elsewhere.
    return stable_hash(sorted(facts, key=stable_hash))


def _duplicate_report(tickets: list[Ticket]) -> dict[str, Any]:
    exact: dict[str, list[Ticket]] = defaultdict(list)
    templates: dict[str, int] = Counter(template_fingerprint(t) for t in tickets)
    for ticket in tickets:
        exact[text_fingerprint(ticket)].append(ticket)
    duplicates = [items for items in exact.values() if len(items) > 1]
    conflicts = sum(
        len({signature for t in items if (signature := _fact_signature(t)) is not None}) > 1
        for items in duplicates
    )
    mixed_coverage = sum(
        any(t.facts is None for t in items) and any(t.facts is not None for t in items)
        for items in duplicates
    )
    _, near = near_duplicate_pairs(tickets)
    return {
        "duplicate_ticket_ids": len(tickets) - len({t.id for t in tickets}),
        "exact_text_groups": len(duplicates),
        "exact_text_excess_records": sum(len(items) - 1 for items in duplicates),
        "template_groups": sum(count > 1 for count in templates.values()),
        "template_excess_records": sum(count - 1 for count in templates.values() if count > 1),
        "conflicting_annotation_groups": conflicts,
        "mixed_label_coverage_exact_groups": mixed_coverage,
        "annotation_identity": "type,value,polarity,temporality,certainty; evidence excluded",
        "near_duplicate_screen": near,
        "automatic_deletion": False,
    }


def analyze(tickets: list[Ticket], tokenizer: Callable | None = None) -> dict[str, Any]:
    """Describe every record in the supplied scope, or refuse an oversized scope.

    Category names are pseudonymized because even labels/roles can contain
    identifiers. The canonical schema is reported; raw-field typing must be
    audited before adaptation. No raw values, IDs, paths or sample texts appear.
    A supplied tokenizer is trusted local code and must count actual model tokens.
    """
    validate_scope(tickets)
    ticket_chars, ticket_words, message_chars, message_words = [], [], [], []
    ticket_tokens, message_tokens, facts_per_labelled = [], [], []
    roles: dict[str, dict[str, Any]] = {}
    fact_types: Counter = Counter()
    ticket_languages: Counter = Counter()
    message_languages: Counter = Counter()
    pattern_tickets: Counter = Counter()
    pattern_messages: Counter = Counter()
    pattern_occurrences: Counter = Counter()
    metadata_types: dict[str, Counter] = defaultdict(Counter)
    timestamp_nulls = 0
    total_messages = 0
    total_facts = 0
    empty_messages = 0
    no_evidence_facts = 0
    qualifier_counts = {"polarity": Counter(), "temporality": Counter(), "certainty": Counter()}
    annotation_duplicates = 0
    for ticket in tickets:
        chars, words, tokens = 0, 0, 0
        detected = set()
        for key, value in ticket.metadata.items():
            metadata_types[_token(str(key))][type(value).__name__] += 1
        for message in ticket.messages:
            text = message.content
            total_messages += 1
            char_count, word_count = len(text), len(text.split())
            chars += char_count
            words += word_count
            message_chars.append(char_count)
            message_words.append(word_count)
            timestamp_nulls += message.timestamp is None
            empty_messages += not text.strip()
            message_languages[_language(text)] += 1
            role = roles.setdefault(
                _token(message.role), {"messages": 0, "characters": 0, "whitespace_units": 0}
            )
            role["messages"] += 1
            role["characters"] += char_count
            role["whitespace_units"] += word_count
            if tokenizer is not None:
                count = _real_token_count(tokenizer, text)
                message_tokens.append(count)
                tokens += count
            for name, pattern in _PATTERNS.items():
                matches = list(pattern.finditer(text))
                if name == "ipv4":
                    valid = 0
                    for match in matches:
                        try:
                            ipaddress.IPv4Address(match.group())
                            valid += 1
                        except ipaddress.AddressValueError:
                            pass
                    count = valid
                else:
                    count = len(matches)
                if count:
                    detected.add(name)
                    pattern_messages[name] += 1
                    pattern_occurrences[name] += count
            mixed_words = sum(
                bool(_CYRILLIC.search(word)) and bool(_LATIN.search(word))
                for word in _WORD.findall(text)
            )
            if mixed_words:
                detected.add("mixed_script_word")
                pattern_messages["mixed_script_word"] += 1
                pattern_occurrences["mixed_script_word"] += mixed_words
        ticket_chars.append(chars)
        ticket_words.append(words)
        ticket_languages[_language("\n".join(m.content for m in ticket.messages))] += 1
        if tokenizer is not None:
            ticket_tokens.append(tokens)
        pattern_tickets.update(detected)
        if ticket.facts is not None:
            facts_per_labelled.append(len(ticket.facts))
            signatures = []
            for fact in ticket.facts:
                total_facts += 1
                fact_types[_token(fact.type)] += 1
                no_evidence_facts += not fact.evidence
                for field, counts in qualifier_counts.items():
                    counts[getattr(fact, field)] += 1
                signatures.append(stable_hash(fact.model_dump(mode="json", exclude={"evidence"})))
            annotation_duplicates += len(signatures) - len(set(signatures))
    labelled = sum(t.facts is not None for t in tickets)
    names = [*_PATTERNS, "mixed_script_word"]
    return {
        "schema_version": "compactfacts.eda.v1",
        "status": "exploratory",
        "scope": {
            "records_requested": len(tickets),
            "records_inspected": len(tickets),
            "all_supplied_records_inspected": True,
            "sampling": "none",
            "record_budget": MAX_TICKETS,
            "text_character_budget": MAX_TEXT_CHARS,
            "input_stage": "validated canonical tickets; raw input typing not observed",
            "ordering": "supplied scope; no representativeness assumption",
        },
        "privacy": {
            "raw_texts_or_ids_emitted": False,
            "category_names": "sha256 pseudonyms",
            "pseudonyms_are_anonymization": False,
        },
        "counts": {
            "tickets": len(tickets),
            "messages": total_messages,
            "labelled_tickets": labelled,
            "unlabelled_tickets": len(tickets) - labelled,
            "labelled_empty_tickets": sum(t.facts == [] for t in tickets),
            "facts": total_facts,
            "empty_messages": empty_messages,
            "tickets_without_messages": sum(not t.messages for t in tickets),
            "facts_without_evidence": no_evidence_facts,
            "duplicate_annotated_facts": annotation_duplicates,
        },
        "schema": {
            "field_types": {
                "ticket.id": {"str": len(tickets)},
                "ticket.messages": {"list": len(tickets)},
                "ticket.facts": {"list": labelled, "null": len(tickets) - labelled},
                "ticket.group_id": {
                    "str": sum(t.group_id is not None for t in tickets),
                    "null": sum(t.group_id is None for t in tickets),
                },
                "message.content": {"str": total_messages},
            },
            "null_counts": {
                "facts": len(tickets) - labelled,
                "group_id": sum(t.group_id is None for t in tickets),
                "message.timestamp": timestamp_nulls,
            },
            "metadata_value_types": {k: dict(v) for k, v in sorted(metadata_types.items())},
        },
        "lengths": {
            "ticket_characters": _summary(ticket_chars),
            "message_characters": _summary(message_chars),
            "ticket_whitespace_units": _summary(ticket_words),
            "message_whitespace_units": _summary(message_words),
            "ticket_model_tokens": _summary(ticket_tokens) if tokenizer is not None else None,
            "message_model_tokens": _summary(message_tokens) if tokenizer is not None else None,
            "tokenizer_supplied": tokenizer is not None,
            "token_count_scope": "sum of message content encodings; excludes chat template",
            "unavailable_token_reason": None
            if tokenizer is not None
            else "real tokenizer not supplied",
            "units": "characters=Unicode code points; whitespace units are not model tokens",
            "outlier_policy": "1.5*IQR flags and mean sensitivity; no exclusion or deletion",
            "facts_per_labelled_ticket": _summary(facts_per_labelled),
        },
        "fact_types": dict(sorted(fact_types.items())),
        "qualifiers": {k: dict(v) for k, v in qualifier_counts.items()},
        "roles": dict(sorted(roles.items())),
        "language_heuristic": {
            "method": "presence of Cyrillic and ASCII Latin letters",
            "tickets": {name: ticket_languages[name] for name in ("ru", "en", "mixed", "unknown")},
            "messages": {
                name: message_languages[name] for name in ("ru", "en", "mixed", "unknown")
            },
            "limitation": "script heuristic, not language identification; identifiers affect it",
        },
        "patterns": {
            name: {
                "tickets": pattern_tickets[name],
                "messages": pattern_messages[name],
                "occurrences": pattern_occurrences[name],
            }
            for name in names
        },
        "duplicates": _duplicate_report(tickets),
        "limitations": [
            "official data dictionary, annotation policy and independent unit remain unknown",
            "support-ticket messages are repeated observations within tickets",
            "group dependencies require supplied metadata and split review",
            "regex patterns and language labels are exploratory diagnostics, not semantic truth",
            "no causal or leaderboard-quality inference follows from these aggregates",
            "near-duplicate completeness is separate from full-scope aggregate coverage",
        ],
        "raw_data_changed": False,
    }
