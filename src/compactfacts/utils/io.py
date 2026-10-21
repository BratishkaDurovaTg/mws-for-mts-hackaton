"""Bounded strict data IO; errors never include raw record contents."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _constant(_):
    raise ValueError("non-finite JSON constant")


def _float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def strict_loads(value: str):
    try:
        return json.loads(
            value, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float
        )
    except (json.JSONDecodeError, RecursionError, OverflowError) as exc:
        raise ValueError("invalid JSON") from exc


def read_jsonl(path, max_bytes=64 * 1024**2, max_records=100000) -> list[dict]:
    path = Path(path)
    if path.suffix not in {".jsonl", ".ndjson"}:
        raise ValueError("expected declared JSONL format")
    if path.is_symlink() or not path.is_file():
        raise ValueError("input must be a regular non-symlink file")
    if path.stat().st_size > max_bytes:
        raise ValueError("input exceeds byte budget")
    records = []
    scanned = 0
    try:
        with path.open(encoding="utf-8-sig", errors="strict") as stream:
            for line_no, line in enumerate(stream, 1):
                scanned += len(line.encode("utf-8"))
                if scanned > max_bytes or len(records) >= max_records:
                    raise ValueError("input exceeds scan budget")
                if not line.strip():
                    raise ValueError(f"empty JSONL record at line {line_no}")
                try:
                    record = strict_loads(line)
                except ValueError:
                    raise ValueError(f"invalid JSONL record at line {line_no}") from None
                if not isinstance(record, dict):
                    raise ValueError(f"JSONL record must be object at line {line_no}")
                records.append(record)
    except UnicodeError:
        raise ValueError("input is not valid UTF-8") from None
    return records


def atomic_text(path, content: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".compactfacts-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, obj):
    atomic_text(path, json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def write_jsonl(path, records):
    atomic_text(
        path, "".join(json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n" for x in records)
    )


def file_hash(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
