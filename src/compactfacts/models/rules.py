"""Provisional exact-mention baseline, with deliberately limited qualifier heuristics.

Does not infer problem causes, service ontology or the latest true dialogue state.
"""

import ipaddress
import re
from datetime import date

from compactfacts.inference import error_prediction
from compactfacts.postprocessing import fact_key
from compactfacts.schemas import Evidence, Fact, InferenceConfig, Prediction, Ticket

PATTERNS = {
    "error_code": re.compile(
        r"(?:ошибк[а-яё]*|error(?:\s+code)?|http(?:\s+status)?|status)\s*[:#=]?\s*"
        r"(?P<value>[45]\d{2}|[A-Z][A-Z0-9_]*_[A-Z0-9_]+)\b",
        re.IGNORECASE,
    ),
    "ip": re.compile(r"(?<![\w.])(?P<value>(?:\d{1,3}\.){3}\d{1,3})(?![\w.])"),
    "uuid": re.compile(r"\b(?P<value>[a-fA-F0-9]{8}-(?:[a-fA-F0-9]{4}-){3}[a-fA-F0-9]{12})\b"),
    "version": re.compile(r"(?<![\w.])(?P<value>v?\d+\.\d+\.\d+(?:[-+][\w.-]+)?)(?![\w.])"),
    "resource_id": re.compile(
        r"\b(?P<value>(?:bucket|vm|instance|resource|ticket|incident)[-_][\w.-]+)", re.IGNORECASE
    ),
    "date": re.compile(r"(?<!\d)(?P<value>20\d{2}-\d{2}-\d{2})(?!\d)"),
    "url": re.compile(r"(?P<value>https?://[^\s<>\"']+)", re.IGNORECASE),
}


def _qualifiers(text, start, end):
    # Clause-local only. Period boundaries require following whitespace to avoid splitting IPs.
    boundaries = list(re.finditer(r"[.!?;]\s+|\n|[,，]\s*(?:но|but|а)\s+", text))
    left = max((m.end() for m in boundaries if m.end() <= start), default=0)
    right = min((m.start() for m in boundaries if m.start() >= end), default=len(text))
    before, after = text[left:start].casefold(), text[end:right].casefold()
    negated = bool(
        re.search(r"(?:\bне\b|\bno\b|\bnot\b|\bwithout\b|\bнет\b)[^,;.!?]{0,24}$", before)
    )
    negated |= bool(
        re.match(r"\s*(?:нет\b|не\s+(?:было|наблюдается)|отсутствует\b|not\s+present\b)", after)
    )
    hypothetical = bool(
        re.search(r"возможно|предполага|думал|кажется|maybe|perhaps|suspect|might", before)
    )
    historical = bool(re.search(r"раньше|ранее|вчера|previously|yesterday|used\s+to", before))
    current = bool(re.search(r"сейчас|теперь|now|currently", before))
    return {
        "polarity": "negated" if negated else "affirmed",
        "temporality": "historical" if historical else "current" if current else "unknown",
        "certainty": "hypothetical" if hypothetical else "asserted",
    }


class RulesExtractor:
    def __init__(self, config: InferenceConfig):
        self.config = config

    def predict(self, ticket: Ticket) -> Prediction:
        if sum(len(m.content) for m in ticket.messages) > self.config.max_input_chars:
            return error_prediction(ticket.id, "input_too_long")
        facts = {}
        for message in ticket.messages:
            for fact_type, pattern in PATTERNS.items():
                if (
                    self.config.allowed_types is not None
                    and fact_type not in self.config.allowed_types
                ):
                    continue
                for match in pattern.finditer(message.content):
                    value = match.group("value")
                    start, end = match.span("value")
                    if fact_type == "ip":
                        try:
                            ipaddress.ip_address(value)
                        except ValueError:
                            continue
                    if fact_type == "date":
                        try:
                            date.fromisoformat(value)
                        except ValueError:
                            continue
                    if fact_type == "url":
                        value = value.rstrip(".,;!?)，。")
                        end = start + len(value)
                    if fact_type == "resource_id":
                        value = value.rstrip(".,;!?)，。")
                        end = start + len(value)
                    evidence = Evidence(message_id=message.id, start=start, end=end, text=value)
                    fact = Fact(
                        type=fact_type,
                        value=value,
                        evidence=[evidence],
                        **_qualifiers(message.content, start, end),
                    )
                    key = fact_key(fact)
                    if key in facts:
                        facts[key].evidence.append(evidence)
                    else:
                        facts[key] = fact
        return Prediction(ticket_id=ticket.id, status="ok", facts=list(facts.values()))
