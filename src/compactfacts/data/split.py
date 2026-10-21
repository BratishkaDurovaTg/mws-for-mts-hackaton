"""Frozen connected-component splits and bounded leakage diagnostics.

Template/near matches are conservative diagnostic links, never deletion rules.
Detection limits are recorded; a negative screen does not establish independence.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from typing import Any

from compactfacts.data.adapters import validate_unique_ids
from compactfacts.schemas import Ticket
from compactfacts.utils.io import stable_hash

MAX_TICKETS = 100_000
MAX_TEXT_CHARS = 64 * 1024 * 1024
MAX_NEAR_PAIRS = 20_000
MAX_SEQUENCE_CHARS = 4_000
MAX_SHINGLES = 2_048
GROUP_FIELDS = ("incident_id", "customer_id", "family_id", "template_id")

_PLACEHOLDERS = (
    (re.compile(r"https?://[^\s<>]+", re.I), "<URL>"),
    (re.compile(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", re.I), "<UUID>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<IP>"),
    (
        re.compile(
            r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"
        ),
        "<TIME>",
    ),
    (re.compile(r"\b[vV]?\d+(?:\.\d+){1,3}(?:[-+][\w.-]+)?\b"), "<VERSION>"),
    (re.compile(r"\b(?:[A-Za-z]+[-_])+\d+[\w-]*\b"), "<RESOURCE>"),
    (re.compile(r"\b\d+\b"), "<NUMBER>"),
)


def validate_scope(tickets: list[Ticket]) -> None:
    if len(tickets) > MAX_TICKETS:
        raise ValueError("ticket count exceeds analysis budget; use an explicit derived scope")
    if sum(len(m.content) for t in tickets for m in t.messages) > MAX_TEXT_CHARS:
        raise ValueError("text exceeds analysis character budget; no records were sampled")


def ticket_text(ticket: Ticket) -> str:
    """Stable comparison text retains message boundaries, ordering and roles."""
    return "\n".join(f"{m.role}\x1f{m.content}" for m in ticket.messages)


def text_fingerprint(ticket: Ticket) -> str:
    return stable_hash([(m.role, m.content) for m in ticket.messages])


def template_fingerprint(ticket: Ticket) -> str:
    parts = []
    for message in ticket.messages:
        text = message.content.casefold()
        for pattern, placeholder in _PLACEHOLDERS:
            text = pattern.sub(placeholder, text)
        parts.append((message.role, " ".join(text.split())))
    return stable_hash(parts)


def _groups(ticket: Ticket) -> list[tuple[str, str]]:
    groups = []
    if ticket.group_id is not None:
        groups.append(("group_id", stable_hash(ticket.group_id)))
    for field in GROUP_FIELDS:
        # Never infer absent group fields. Empty/null values do not assert a link.
        if field in ticket.metadata and ticket.metadata[field] not in (None, ""):
            value = ticket.metadata[field]
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError("declared grouping metadata must be scalar")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("declared grouping metadata must be finite")
            groups.append((field, stable_hash(value)))
    return groups


def _shingles(text: str) -> tuple[set[str], bool]:
    words = text.casefold().split()
    if len(words) < 3:
        values = {text.casefold()[i : i + 4] for i in range(max(1, len(text) - 3))}
    else:
        values = {" ".join(words[i : i + 3]) for i in range(len(words) - 2)}
    limited = len(values) > MAX_SHINGLES
    # Deterministic bottom-k is a bounded approximation, reported explicitly.
    return (set(sorted(values, key=stable_hash)[:MAX_SHINGLES]) if limited else values, limited)


def near_duplicate_pairs(
    tickets: list[Ticket],
    threshold: float = 0.9,
    *,
    max_pairs: int = MAX_NEAR_PAIRS,
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """Compare all pairs for small scopes, otherwise a bounded lexical window.

    Similarity is max(full/prefix SequenceMatcher ratio, word-three-shingle
    Jaccard). Both paths and all approximation limits are disclosed. A lexical
    window is a candidate heuristic and can miss near duplicates.
    """
    if not 0 < threshold <= 1 or not math.isfinite(threshold):
        raise ValueError("near threshold must be finite in (0,1]")
    if not isinstance(max_pairs, int) or isinstance(max_pairs, bool) or max_pairs < 0:
        raise ValueError("max_pairs must be a non-negative integer")
    ordered = sorted(tickets, key=lambda ticket: ticket.id)
    texts = {t.id: ticket_text(t) for t in ordered}
    n = len(ordered)
    total_pairs = n * (n - 1) // 2
    all_pairs = total_pairs <= max_pairs
    if all_pairs:
        candidates = [(a.id, b.id) for a, b in combinations(ordered, 2)]
    else:
        # Sorting is O(N log N); at most 8N candidates, capped during creation.
        lexical = sorted(ordered, key=lambda t: (texts[t.id].casefold(), t.id))
        candidates = []
        for i, ticket in enumerate(lexical):
            for other in lexical[i + 1 : i + 9]:
                if len(candidates) >= max_pairs:
                    break
                candidates.append(tuple(sorted((ticket.id, other.id))))
            if len(candidates) >= max_pairs:
                break
    signatures: dict[str, set[str]] = {}
    shingle_limited = set()
    sequence_limited = {key for key, text in texts.items() if len(text) > MAX_SEQUENCE_CHARS}
    matches = []
    for left, right in candidates:
        for key in (left, right):
            if key not in signatures:
                signatures[key], limited = _shingles(texts[key])
                if limited:
                    shingle_limited.add(key)
        a, b = signatures[left], signatures[right]
        jaccard = len(a & b) / len(a | b) if a or b else 1.0
        sequence = SequenceMatcher(
            None,
            texts[left][:MAX_SEQUENCE_CHARS],
            texts[right][:MAX_SEQUENCE_CHARS],
            autojunk=True,
        ).ratio()
        # Prefix equality must not imply full-record equality for long texts.
        if left in sequence_limited or right in sequence_limited:
            sequence = 0.0
        if max(jaccard, sequence) >= threshold:
            matches.append((left, right))
    report = {
        "method": "max_sequence_ratio_word3_jaccard",
        "candidate_method": "all_pairs" if all_pairs else "lexical_window_8",
        "threshold": threshold,
        "pair_budget": max_pairs,
        "possible_pairs": total_pairs,
        "compared_pairs": len(candidates),
        "detected_pairs": len(matches),
        "candidate_coverage_complete": all_pairs,
        "comparison_complete": all_pairs and not sequence_limited and not shingle_limited,
        "sequence_char_limit": MAX_SEQUENCE_CHARS,
        "sequence_limited_records": len(sequence_limited),
        "shingle_limit": MAX_SHINGLES,
        "shingle_limited_records": len(shingle_limited),
        "limitations": [
            "near similarity is a heuristic, not semantic duplicate adjudication",
            "bounded candidate windows can miss links when candidate_coverage_complete=false",
            "long-record comparison uses bounded shingles; no sequence prefix equality shortcut",
        ],
    }
    return sorted(matches), report


class _Components:
    def __init__(self, ids: list[str]):
        self.parents = {key: key for key in ids}

    def find(self, key: str) -> str:
        while self.parents[key] != key:
            self.parents[key] = self.parents[self.parents[key]]
            key = self.parents[key]
        return key

    def join(self, left: str, right: str) -> None:
        a, b = sorted((self.find(left), self.find(right)))
        self.parents[b] = a


def leakage_audit(
    train: list[Ticket],
    dev: list[Ticket],
    near_threshold: float = 0.9,
    *,
    max_pairs: int = MAX_NEAR_PAIRS,
) -> dict[str, Any]:
    """Aggregate screen; no raw IDs, customer values or support text are emitted."""
    validate_scope(train + dev)
    left_ids, right_ids = {t.id for t in train}, {t.id for t in dev}

    def overlap(function):
        return len({function(t) for t in train} & {function(t) for t in dev})

    left_groups = {g for t in train for g in _groups(t)}
    right_groups = {g for t in dev for g in _groups(t)}
    groups = {
        name: sum(1 for field, _ in left_groups & right_groups if field == name)
        for name in ("group_id", *GROUP_FIELDS)
    }
    pairs, near = near_duplicate_pairs(train + dev, near_threshold, max_pairs=max_pairs)
    crossing = sum(
        (a in left_ids and b in right_ids) or (b in left_ids and a in right_ids) for a, b in pairs
    )
    result = {
        "ticket_id_overlap": len(left_ids & right_ids),
        "group_overlaps": groups,
        "exact_text_overlap": overlap(text_fingerprint),
        "template_overlap": overlap(template_fingerprint),
        "detected_near_cross_pairs": crossing,
        "near_duplicate_screen": near,
        "interpretation": "not detected only within declared scope and comparison limits",
    }
    result["passed"] = not (
        result["ticket_id_overlap"]
        or any(groups.values())
        or result["exact_text_overlap"]
        or result["template_overlap"]
        or crossing
    )
    return result


def split_tickets(
    tickets: list[Ticket],
    dev_fraction: float = 0.25,
    seed: int = 42,
    near_threshold: float = 0.9,
) -> tuple[list[Ticket], list[Ticket], dict[str, Any]]:
    """Keep every record; deterministic grouping is independent of input order.

    The controlled frozen manifest contains IDs for exact replay. EDA/logging
    should use hashes/aggregate results instead. Keep this manifest private with
    the source data. Its dataset hash includes annotations and metadata.
    """
    if not 0 < dev_fraction < 1 or not math.isfinite(dev_fraction):
        raise ValueError("dev_fraction must be finite in (0,1)")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    ordered = sorted(validate_unique_ids(tickets), key=lambda t: t.id)
    validate_scope(ordered)
    components = _Components([t.id for t in ordered])
    seen: dict[tuple[str, str], str] = {}
    for ticket in ordered:
        keys = [
            ("exact", text_fingerprint(ticket)),
            ("template", template_fingerprint(ticket)),
            *_groups(ticket),
        ]
        for key in keys:
            if key in seen:
                components.join(ticket.id, seen[key])
            else:
                seen[key] = ticket.id
    pairs, near = near_duplicate_pairs(ordered, near_threshold)
    for left, right in pairs:
        components.join(left, right)
    groups: dict[str, list[str]] = defaultdict(list)
    for ticket in ordered:
        groups[components.find(ticket.id)].append(ticket.id)
    if len(groups) < 2:
        raise ValueError("split requires at least two independent detected components")
    group_order = sorted(groups.values(), key=lambda ids: stable_hash([seed, ids]))
    desired = max(1, min(len(ordered) - 1, round(len(ordered) * dev_fraction)))
    dev_ids: set[str] = set()
    for ids in group_order:
        # Keep at least one component in train and choose nearest achievable size.
        if len(dev_ids) + len(ids) == len(ordered):
            continue
        before = abs(len(dev_ids) - desired)
        after = abs(len(dev_ids) + len(ids) - desired)
        if after < before:
            dev_ids.update(ids)
    if not dev_ids:
        chosen = min(
            group_order, key=lambda ids: (abs(len(ids) - desired), stable_hash([seed, ids]))
        )
        dev_ids.update(chosen)
    train = [t for t in ordered if t.id not in dev_ids]
    dev = [t for t in ordered if t.id in dev_ids]
    audit = leakage_audit(train, dev, near_threshold)
    if not audit["passed"]:
        raise ValueError("split failed its leakage screen; no split was emitted")
    manifest: dict[str, Any] = {
        "schema_version": "compactfacts.split.v1",
        "dataset_hash": stable_hash([t.model_dump(mode="json") for t in ordered]),
        "seed": seed,
        "requested_dev_fraction": dev_fraction,
        "actual_dev_fraction": len(dev) / len(ordered),
        "near_threshold": near_threshold,
        "group_fields": ["group_id", *[f"metadata.{f}" for f in GROUP_FIELDS]],
        "group_fields_policy": "join only explicitly provided non-null scalar values",
        "connected_components": len(groups),
        "component_sizes": sorted(len(ids) for ids in groups.values()),
        "train_ids": [t.id for t in train],
        "dev_ids": [t.id for t in dev],
        "ticket_hashes": {t.id: stable_hash(t.model_dump(mode="json")) for t in ordered},
        "near_duplicate_screen": near,
        "leakage_audit": audit,
        "privacy": "controlled manifest contains raw IDs; keep private with source data",
        "limitations": [
            "target fraction can be unattainable without splitting a detected component",
            "unobserved dependencies and missed near duplicates can remain",
            "template and near links are conservative heuristics, not annotation judgments",
        ],
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    return train, dev, manifest
