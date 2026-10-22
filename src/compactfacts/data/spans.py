"""Exact-copy boundary shared by span/tagger outputs and public IE adapters.

This converts predicted spans; it does not predict, classify, link or normalize them.
"""

from compactfacts.schemas import Evidence, Fact, Ticket


def fact_from_span(
    ticket: Ticket, message_id: str, start: int, end: int, fact_type: str, **qualifiers
) -> Fact:
    message = next((m for m in ticket.messages if m.id == message_id), None)
    if message is None:
        raise ValueError("unknown source message")
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= len(message.content)
    ):
        raise ValueError("span outside original message")
    value = message.content[start:end]
    return Fact(
        type=fact_type,
        value=value,
        evidence=[Evidence(message_id=message_id, start=start, end=end, text=value)],
        **qualifiers,
    )
