"""Append-only research ledger; intentionally no external experiment service."""

import fcntl
import json
import os
from pathlib import Path


def append_run(path, record):
    required = {
        "id",
        "timestamp_utc",
        "git_commit",
        "hypothesis",
        "change",
        "expected_result",
        "decision_rule",
        "data_sha256",
        "config",
        "command",
        "seed",
        "metrics",
        "benchmark",
        "artifact",
        "conclusion",
        "decision",
        "reason",
    }
    if not required <= record.keys() or record["decision"] not in {"KEEP", "REJECT"}:
        raise ValueError("incomplete experiment ledger record")
    if any(
        not isinstance(record[key], str) or not record[key].strip()
        for key in ["hypothesis", "change", "expected_result", "decision_rule"]
    ):
        raise ValueError("experiment requires four nonempty hypothesis/change/expected/rule fields")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # One locked append, no overwrite/rewrite of previous experiments.
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        last_line = ""
        for line in stream:
            if json.loads(line)["id"] == record["id"]:
                raise ValueError("experiment id already exists")
            last_line = line
        if last_line and not last_line.endswith("\n"):
            raise ValueError("existing ledger has incomplete final line")
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
