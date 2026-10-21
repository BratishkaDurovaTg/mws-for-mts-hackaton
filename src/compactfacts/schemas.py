"""Provisional internal schema, deliberately independent of the unknown MWS format."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonEmpty = Annotated[str, Field(min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class Message(StrictModel):
    id: NonEmpty
    role: NonEmpty
    content: str
    timestamp: str | None = None


class Evidence(StrictModel):
    message_id: NonEmpty
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: NonEmpty

    @model_validator(mode="after")
    def valid_span(self) -> Self:
        if self.end <= self.start or self.end - self.start != len(self.text):
            raise ValueError("invalid evidence span length")
        return self


class Fact(StrictModel):
    type: NonEmpty
    value: NonEmpty
    evidence: list[Evidence] = Field(default_factory=list)
    polarity: Literal["affirmed", "negated"] = "affirmed"
    temporality: Literal["current", "historical", "unknown"] = "unknown"
    certainty: Literal["asserted", "hypothetical"] = "asserted"


class Ticket(StrictModel):
    id: NonEmpty
    messages: list[Message]
    facts: list[Fact] | None = None
    group_id: NonEmpty | None = None
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_messages(self) -> Self:
        ids = [m.id for m in self.messages]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate message id")
        return self


class FactPayload(StrictModel):
    facts: list[Fact]


class Prediction(StrictModel):
    ticket_id: NonEmpty
    status: Literal["ok", "error"]
    facts: list[Fact]
    error_code: NonEmpty | None = None
    repaired: bool = False
    generated_tokens: int | None = Field(default=None, ge=0)
    attempts: int = Field(default=1, ge=1, le=3)

    @model_validator(mode="after")
    def consistent_status(self) -> Self:
        if self.status == "error" and (self.facts or self.error_code is None):
            raise ValueError("error prediction requires code and no facts")
        if self.status == "ok" and self.error_code is not None:
            raise ValueError("successful prediction cannot carry error")
        return self


class InferenceConfig(StrictModel):
    backend: Literal["empty", "rules", "hf"] = "rules"
    model_path: str | None = None
    model_id: str | None = None
    revision: str | None = None
    adapter_path: str | None = None
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    max_input_tokens: int = Field(default=4096, gt=0)
    max_new_tokens: int = Field(default=1024, gt=0)
    max_input_chars: int = Field(default=100000, gt=0)
    seed: int = 42
    allowed_types: list[NonEmpty] | None = None
    parser_mode: Literal["strict", "fence"] = "strict"
    max_retries: int = Field(default=0, ge=0, le=2)
    enable_thinking: bool = False
    require_evidence: bool = True

    @model_validator(mode="after")
    def local_pinned_model(self) -> Self:
        if self.backend == "hf":
            import re

            if not self.model_path or not self.model_id:
                raise ValueError("HF backend requires local model_path and model_id provenance")
            if self.revision is None or not re.fullmatch(r"[a-fA-F0-9]{40}", self.revision):
                raise ValueError("HF backend requires immutable 40-character revision")
            if self.enable_thinking:
                raise ValueError("thinking must be disabled for this JSON extraction experiment")
        return self
