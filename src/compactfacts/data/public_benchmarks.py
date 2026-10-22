"""Small pinned public diagnostic data; no model downloads or benchmark-quality claims."""

import argparse
import hashlib
import json
import ssl
import urllib.error
import urllib.request
from pathlib import Path

from compactfacts.schemas import Evidence, Fact, Message, Ticket
from compactfacts.utils.io import atomic_text, strict_loads

REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE_PATH = "nlupp/data/banking/fold0.json"
BASE_URL = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{REVISION}"
SOURCE_URL = f"{BASE_URL}/{SOURCE_PATH}"
LICENSE_URL = f"{BASE_URL}/LICENSE"
SOURCE_SHA256 = "bc9544ba197f5f073d07beab056466997ad986077d1ee46b434a251568034d0c"
LICENSE_SHA256 = "7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661"
MAX_SOURCE_BYTES = 256 * 1024
MAX_RECORDS = 20


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def _verified_bytes(raw: bytes, expected_sha256: str) -> bytes:
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("public source exceeds byte budget")
    if _sha256(raw) != expected_sha256:
        raise ValueError("public source SHA256 mismatch")
    return raw


def read_pinned_file(path: Path, expected_sha256: str) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("public source must be a regular non-symlink file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_SOURCE_BYTES + 1)
    return _verified_bytes(raw, expected_sha256)


def download_pinned(url: str, expected_sha256: str) -> bytes:
    """Fetch only known immutable resources, retaining normal HTTPS verification."""
    if (url, expected_sha256) not in {
        (SOURCE_URL, SOURCE_SHA256),
        (LICENSE_URL, LICENSE_SHA256),
    }:
        raise ValueError("unrecognised pinned public resource")

    def fetch(context):
        with urllib.request.urlopen(url, timeout=20, context=context) as response:
            if response.geturl() != url:
                raise ValueError("unexpected public source redirect")
            raw = response.read(MAX_SOURCE_BYTES + 1)
        return _verified_bytes(raw, expected_sha256)

    try:
        return fetch(ssl.create_default_context())
    except urllib.error.URLError as exc:
        if not isinstance(exc.reason, ssl.SSLCertVerificationError):
            raise
        # Some macOS Python builds do not locate the system CA bundle by default.
        # This still verifies both the certificate chain and the hostname.
        cafile = Path("/etc/ssl/cert.pem")
        if not cafile.is_file():
            raise
        return fetch(ssl.create_default_context(cafile=str(cafile)))


def nlupp_to_ticket(row: dict, index: int) -> Ticket:
    """Preserve upstream slots as exact spans; index is our locator, not an upstream ID."""
    if type(index) is not int or index < 0:
        raise ValueError("invalid source row index")
    if not isinstance(row, dict) or set(row) - {"text", "intents", "slots"}:
        raise ValueError("invalid NLU++ record fields")
    text = row.get("text")
    intents = row.get("intents", [])
    slots = row.get("slots", {})
    if not isinstance(text, str) or not text:
        raise ValueError("NLU++ record requires nonempty text")
    if not isinstance(intents, list) or any(not isinstance(x, str) or not x for x in intents):
        raise ValueError("invalid NLU++ intent annotations")
    if not isinstance(slots, dict):
        raise ValueError("invalid NLU++ slot annotations")
    facts = []
    for name, slot in slots.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(slot, dict):
            raise ValueError("invalid NLU++ slot")
        if set(slot) - {"text", "span", "value", "values"}:
            raise ValueError("unexpected NLU++ slot fields")
        literal, span = slot.get("text"), slot.get("span")
        if not isinstance(literal, str) or not literal:
            raise ValueError("invalid NLU++ slot literal")
        if not isinstance(span, list) or len(span) != 2 or any(type(x) is not int for x in span):
            raise ValueError("invalid NLU++ span coordinates")
        start, end = span
        if not 0 <= start < end <= len(text) or text[start:end] != literal:
            raise ValueError("NLU++ span does not match original text")
        facts.append(
            Fact(
                type=f"nlupp.slot.{name}",
                value=literal,
                evidence=[Evidence(message_id="utterance", start=start, end=end, text=literal)],
            )
        )
    # Source annotations (including structured canonical values) are never prompts.
    # Qualifier defaults are transport placeholders, not NLU++ semantic labels.
    return Ticket(
        id=f"nlupp:{SOURCE_PATH}:{index}",
        messages=[Message(id="utterance", role="user", content=text)],
        facts=facts,
        group_id=f"nlupp:{REVISION}:{SOURCE_PATH}",
        metadata={
            "synthetic": False,
            "public_benchmark": True,
            "dataset": "PolyAI NLU++",
            "source_revision": REVISION,
            "source_path": SOURCE_PATH,
            "source_index": index,
            "id_is_source_locator": True,
            "source_record": row,
            "source_role_field_present": False,
            "role_basis": "dataset describes user utterances",
            "qualifier_labels_available": False,
        },
    )


def prepare_benchmark(
    output: Path,
    count: int = 10,
    source_file: Path | None = None,
    license_file: Path | None = None,
) -> dict:
    """Create a deterministic evaluation-only slice without overwriting prior artifacts."""
    if type(count) is not int or not 1 <= count <= MAX_RECORDS:
        raise ValueError("public benchmark count must be between 1 and 20")
    if (source_file is None) != (license_file is None):
        raise ValueError("offline preparation requires both source and license files")
    output = Path(output)
    names = ("source.json", "LICENSE.txt", "tickets.jsonl", "manifest.json")
    if output.is_symlink() or any((output / name).exists() for name in names):
        raise ValueError("public benchmark output already exists or is a symlink")
    source = (
        read_pinned_file(source_file, SOURCE_SHA256)
        if source_file is not None
        else download_pinned(SOURCE_URL, SOURCE_SHA256)
    )
    license_raw = (
        read_pinned_file(license_file, LICENSE_SHA256)
        if license_file is not None
        else download_pinned(LICENSE_URL, LICENSE_SHA256)
    )
    try:
        rows = strict_loads(source.decode("utf-8"))
    except UnicodeError:
        raise ValueError("public source is not valid UTF-8") from None
    if not isinstance(rows, list) or len(rows) != 104:
        raise ValueError("pinned NLU++ source record count mismatch")
    tickets = [nlupp_to_ticket(row, index) for index, row in enumerate(rows[:count])]
    canonical_rows = [_json_bytes(ticket.model_dump(mode="json")) for ticket in tickets]
    canonical = b"".join(row + b"\n" for row in canonical_rows)
    manifest = {
        "schema_version": 1,
        "purpose": "public auxiliary diagnostic evaluation, not MWS quality estimation",
        "synthetic": False,
        "dataset": "PolyAI NLU++",
        "source_url": SOURCE_URL,
        "source_revision": REVISION,
        "source_path": SOURCE_PATH,
        "source_sha256": SOURCE_SHA256,
        "source_bytes": len(source),
        "license": "CC-BY-4.0",
        "license_url": LICENSE_URL,
        "license_sha256": LICENSE_SHA256,
        "attribution": (
            "PolyAI Limited; Iñigo Casanueva, Ivan Vulić, Georgios P. Spithourakis, "
            "Paweł Budzianowski (2022), NLU++"
        ),
        "paper": "https://arxiv.org/abs/2204.13021",
        "changes": (
            "first rows selected; literal slots converted to canonical tickets; no relabelling"
        ),
        "source_record_count": len(rows),
        "record_count": count,
        "selection": "first count records by zero-based upstream array position",
        "split": "frozen auxiliary eval; source fold is not an official fixed test split",
        "exclude_from_future_training": f"{REVISION}:{SOURCE_PATH}",
        "canonical_file": "tickets.jsonl",
        "canonical_sha256": _sha256(canonical),
        "canonical_bytes": len(canonical),
        "row_hash_serialization": "UTF-8 JSON, sorted keys, ensure_ascii=false; no final newline",
        "records": [
            {
                "source_index": index,
                "ticket_id": ticket.id,
                "source_record_sha256": _sha256(_json_bytes(rows[index])),
                "canonical_record_sha256": _sha256(canonical_rows[index]),
            }
            for index, ticket in enumerate(tickets)
        ],
        "ru_slice_status": "pending: bounded MASSIVE rows endpoint returned Not found",
        "limitations": [
            "English single-turn banking, not technical support",
            "anonymised real examples and expert-created rare cases; origins not separated per row",
            "no negation, temporality, relation, long-context or technical-identifier gold added",
            "qualifier defaults are transport placeholders, not annotated semantics",
            "upstream pretraining contamination is unknown",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(output / "source.json", source.decode("utf-8"))
    atomic_text(output / "LICENSE.txt", license_raw.decode("utf-8"))
    atomic_text(output / "tickets.jsonl", canonical.decode("utf-8"))
    atomic_text(output / "manifest.json", _json_bytes(manifest).decode("utf-8") + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/public_benchmark"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--license-file", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = prepare_benchmark(args.output, args.count, args.source_file, args.license_file)
    except (ValueError, OSError, urllib.error.URLError):
        parser.exit(2, "public benchmark preparation failed; no successful empty benchmark\n")
    print(
        json.dumps(
            {
                "record_count": manifest["record_count"],
                "canonical_sha256": manifest["canonical_sha256"],
                "output": str(args.output),
            }
        )
    )
    return 0
