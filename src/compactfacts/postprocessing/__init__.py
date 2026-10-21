"""Explicit normalization policies; raw identifiers are unchanged by default."""

from collections.abc import Mapping

from compactfacts.schemas import Fact


def normalize_value(value: str, fact_type: str, policy: Mapping[str, str] | None = None) -> str:
    mode = (policy or {}).get(fact_type, "identity")
    if mode == "identity":
        return value
    if mode == "strip":
        return value.strip()
    if mode == "casefold":
        return value.casefold()
    if mode == "whitespace":
        return " ".join(value.split())
    raise ValueError("unknown normalization policy")


def fact_key(fact: Fact, normalization=None) -> tuple[str, ...]:
    return (
        fact.type,
        normalize_value(fact.value, fact.type, normalization),
        fact.polarity,
        fact.temporality,
        fact.certainty,
    )
