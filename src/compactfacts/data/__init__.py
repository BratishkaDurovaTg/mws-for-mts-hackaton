"""Explicit adapters, bounded aggregate EDA and reproducible group splits."""

from compactfacts.data.adapters import (
    CanonicalAdapter,
    MappingAdapter,
    OfficialSubmissionAdapter,
    export_submission,
    read_tickets,
    write_tickets,
)
from compactfacts.data.eda import analyze
from compactfacts.data.split import leakage_audit, split_tickets

__all__ = [
    "CanonicalAdapter",
    "MappingAdapter",
    "OfficialSubmissionAdapter",
    "export_submission",
    "read_tickets",
    "write_tickets",
    "analyze",
    "leakage_audit",
    "split_tickets",
]
