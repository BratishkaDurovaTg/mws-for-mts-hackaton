"""Strict JSON parser, with a separately observable opt-in fence repair."""

import re

from pydantic import ValidationError

from compactfacts.schemas import FactPayload
from compactfacts.utils.io import strict_loads


class ParseFailure(ValueError):
    pass


def parse_payload(text: str, mode: str = "strict") -> tuple[FactPayload, bool]:
    if mode not in {"strict", "fence"}:
        raise ValueError("unknown parser mode")
    if len(text) > 2 * 1024**2:
        raise ParseFailure("output_limit")
    repaired = False
    if mode == "fence":
        match = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", text, re.DOTALL)
        if match:
            text, repaired = match.group(1), True
    try:
        value = strict_loads(text)
    except ValueError:
        raise ParseFailure("invalid_json") from None
    try:
        return FactPayload.model_validate(value), repaired
    except ValidationError:
        raise ParseFailure("invalid_schema") from None
