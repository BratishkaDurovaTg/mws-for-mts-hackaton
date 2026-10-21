"""Reproducible local run metadata without customer text or secrets."""

import importlib.metadata
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from compactfacts.utils.io import file_hash, stable_hash


def provenance() -> dict:
    root = Path.cwd()
    package_root = Path(__file__).resolve().parents[1]
    source_hash = stable_hash(
        {str(p.relative_to(package_root)): file_hash(p) for p in sorted(package_root.rglob("*.py"))}
    )

    def git(*args):
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    versions = {}
    for package in [
        "mws-compactfacts",
        "pydantic",
        "psutil",
        "torch",
        "transformers",
        "peft",
        "accelerate",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    working_tree_status = git("status", "--porcelain")
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(working_tree_status) if working_tree_status is not None else None,
        "package_source_sha256": source_hash,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "versions": versions,
    }
