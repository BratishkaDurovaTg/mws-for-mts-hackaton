"""Replaceable extractor backends with a shared canonical prediction contract."""

from typing import Protocol

from compactfacts.schemas import InferenceConfig, Prediction, Ticket


class Extractor(Protocol):
    def predict(self, ticket: Ticket) -> Prediction: ...


class EmptyExtractor:
    def predict(self, ticket: Ticket) -> Prediction:
        return Prediction(ticket_id=ticket.id, status="ok", facts=[])


def create_extractor(config: InferenceConfig) -> Extractor:
    if config.backend == "empty":
        return EmptyExtractor()
    if config.backend == "rules":
        from compactfacts.models.rules import RulesExtractor

        return RulesExtractor(config)
    if config.backend == "hf":
        from compactfacts.models.hf import HFExtractor

        return HFExtractor(config)
    raise ValueError("unknown backend")
