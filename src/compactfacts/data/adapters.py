"""Explicit external boundaries. No official MWS schema is assumed here."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

from compactfacts.schemas import Prediction, Ticket
from compactfacts.utils.io import read_jsonl, write_jsonl


class TicketAdapter(Protocol):
    def to_ticket(self, record: dict[str, Any]) -> Ticket: ...

    def from_ticket(self, ticket: Ticket) -> dict[str, Any]: ...


class CanonicalAdapter:
    """Strict provisional JSONL representation, including null versus empty facts."""

    def to_ticket(self, record: dict[str, Any]) -> Ticket:
        return Ticket.model_validate(record)

    def from_ticket(self, ticket: Ticket) -> dict[str, Any]:
        return ticket.model_dump(mode="json")


class MappingAdapter:
    """Map declared raw field names to canonical fields without schema guessing.

    ``mapping={"id": "case", "text": "body", "facts": "annotations"}``
    describes single-message records. For conversations, replace ``text`` with
    ``messages`` and declare ``message_mapping`` if inner names differ. Facts
    already use the canonical representation unless ``fact_mapping`` is supplied.
    Unknown raw fields are ignored deliberately; declare ``metadata`` to retain
    an explicitly selected metadata object. The raw input itself is never edited.
    """

    _fields = {"id", "text", "messages", "facts", "group_id", "metadata"}
    _message_fields = {"id", "role", "content", "timestamp"}
    _fact_fields = {"type", "value", "evidence", "polarity", "temporality", "certainty"}

    def __init__(
        self,
        mapping: Mapping[str, str],
        *,
        message_mapping: Mapping[str, str] | None = None,
        fact_mapping: Mapping[str, str] | None = None,
        default_role: str = "user",
    ) -> None:
        self.mapping = self._validate_mapping(mapping, self._fields)
        if "id" not in self.mapping:
            raise ValueError("mapping must declare id")
        if ("text" in self.mapping) == ("messages" in self.mapping):
            raise ValueError("mapping must declare exactly one of text or messages")
        self.message_mapping = (
            self._validate_mapping(message_mapping, self._message_fields)
            if message_mapping is not None
            else None
        )
        self.fact_mapping = (
            self._validate_mapping(fact_mapping, self._fact_fields)
            if fact_mapping is not None
            else None
        )
        if not isinstance(default_role, str) or not default_role.strip():
            raise ValueError("default_role must be a non-empty string")
        self.default_role = default_role

    @staticmethod
    def _validate_mapping(mapping: Mapping[str, str], allowed: set[str]) -> dict[str, str]:
        if not isinstance(mapping, Mapping) or not set(mapping) <= allowed:
            raise ValueError("mapping contains unsupported canonical fields")
        if any(not isinstance(v, str) or not v for v in mapping.values()):
            raise ValueError("raw field names must be non-empty strings")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("raw field names must map one-to-one")
        return dict(mapping)

    @staticmethod
    def _map_item(item: Any, mapping: dict[str, str] | None) -> dict[str, Any]:
        if not isinstance(item, dict):
            raise ValueError("mapped message/fact must be an object")
        if mapping is None:
            return dict(item)
        return {canonical: item[raw] for canonical, raw in mapping.items() if raw in item}

    def to_ticket(self, record: dict[str, Any]) -> Ticket:
        if not isinstance(record, dict):
            raise ValueError("input record must be an object")
        required = ["id", "text" if "text" in self.mapping else "messages"]
        if any(self.mapping[name] not in record for name in required):
            raise ValueError("record is missing a declared required field")
        result = {
            name: record[raw]
            for name, raw in self.mapping.items()
            if name not in {"text", "messages", "facts"} and raw in record
        }
        if "text" in self.mapping:
            result["messages"] = [
                {"id": "m0", "role": self.default_role, "content": record[self.mapping["text"]]}
            ]
        else:
            raw_messages = record[self.mapping["messages"]]
            if not isinstance(raw_messages, list):
                raise ValueError("mapped messages must be an array")
            result["messages"] = [self._map_item(m, self.message_mapping) for m in raw_messages]
        raw_facts = record.get(self.mapping["facts"]) if "facts" in self.mapping else None
        if raw_facts is not None and not isinstance(raw_facts, list):
            raise ValueError("mapped facts must be null or an array")
        result["facts"] = (
            None if raw_facts is None else [self._map_item(f, self.fact_mapping) for f in raw_facts]
        )
        return Ticket.model_validate(result)

    def from_ticket(self, ticket: Ticket) -> dict[str, Any]:
        raise NotImplementedError("raw export requires a separately specified output adapter")


def validate_unique_ids(tickets: Iterable[Ticket]) -> list[Ticket]:
    result = list(tickets)
    seen: set[str] = set()
    for index, ticket in enumerate(result):
        if ticket.id in seen:
            raise ValueError(f"duplicate ticket id at record {index + 1}")
        seen.add(ticket.id)
    return result


def read_tickets(
    path: str | Path,
    adapter: TicketAdapter | None = None,
    *,
    max_bytes: int = 64 * 1024 * 1024,
    max_records: int = 100_000,
) -> list[Ticket]:
    boundary = adapter or CanonicalAdapter()
    records = read_jsonl(path, max_bytes=max_bytes, max_records=max_records)
    tickets: list[Ticket] = []
    for index, record in enumerate(records):
        try:
            tickets.append(boundary.to_ticket(record))
        except (ValueError, TypeError, KeyError):
            # Pydantic diagnostics may contain raw text, so do not echo them.
            raise ValueError(f"invalid ticket at record {index + 1}") from None
    return validate_unique_ids(tickets)


def write_tickets(path: str | Path, tickets: Iterable[Ticket]) -> None:
    records = validate_unique_ids(tickets)
    write_jsonl(path, [ticket.model_dump(mode="json") for ticket in records])


class SubmissionAdapter(Protocol):
    def convert(self, predictions: list[Prediction]) -> list[dict[str, Any]]: ...


class OfficialSubmissionAdapter:
    """Intentional fail-closed placeholder until an official specification exists."""

    def convert(self, predictions: list[Prediction]) -> list[dict[str, Any]]:
        raise NotImplementedError("official submission schema has not been supplied")


def export_submission(
    path: str | Path,
    predictions: Iterable[Prediction],
    *,
    adapter: SubmissionAdapter | None = None,
    official: bool = False,
) -> None:
    """Write canonical diagnostic envelopes or an explicitly supplied adapter.

    Canonical errors stay visible. This diagnostic format is not an official
    competition submission. An official export without an adapter fails closed.
    """
    items = list(predictions)
    if len({p.ticket_id for p in items}) != len(items):
        raise ValueError("duplicate prediction ticket ids")
    boundary = adapter or (OfficialSubmissionAdapter() if official else None)
    converted = (
        boundary.convert(items)
        if boundary is not None
        else [p.model_dump(mode="json") for p in items]
    )
    write_jsonl(path, converted)
