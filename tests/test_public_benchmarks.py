"""Synthetic-only boundary tests: no public text, network, weights or ML extra."""

import copy
import hashlib
import io
import json
import ssl
import urllib.error

import pytest

from compactfacts.data import public_benchmarks as public
from compactfacts.models.prompts import build_messages


@pytest.fixture
def synthetic_row():
    text = "🔧 сбой 007 после 09:30"
    start = text.index("007")
    return {
        "text": text,
        "intents": ["private_gold_intent_marker"],
        "slots": {
            "number": {
                "text": "007",
                "span": [start, start + 3],
                "value": {"number": 7, "gold_only_marker": True},
            }
        },
    }


def test_conversion_copies_literal_unicode_span_and_preserves_raw_annotations(synthetic_row):
    ticket = public.nlupp_to_ticket(synthetic_row, 3)
    fact = ticket.facts[0]
    assert fact.type == "nlupp.slot.number"
    assert fact.value == "007"
    evidence = fact.evidence[0]
    assert ticket.messages[0].content[evidence.start : evidence.end] == "007"
    assert ticket.metadata["source_record"] == synthetic_row
    assert ticket.metadata["synthetic"] is False
    assert ticket.metadata["qualifier_labels_available"] is False
    assert ticket.metadata["source_index"] == 3
    assert ticket.id.endswith(":3")
    assert public.nlupp_to_ticket(synthetic_row, 4).group_id == ticket.group_id
    prompt = json.dumps(build_messages(ticket), ensure_ascii=False)
    assert "private_gold_intent_marker" not in prompt
    assert "gold_only_marker" not in prompt
    assert "nlupp.slot.number" not in prompt


def test_missing_slots_is_labelled_empty_by_upstream_policy():
    assert public.nlupp_to_ticket({"text": "synthetic empty"}, 0).facts == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(slots=None),
        lambda r: r.update(intents="not a list"),
        lambda r: r.update(unexpected="ignored gold would be a defect"),
        lambda r: r["slots"]["number"].update(span=[True, 4]),
        lambda r: r["slots"]["number"].update(span=[-1, 2]),
        lambda r: r["slots"]["number"].update(span=[0, 10000]),
        lambda r: r["slots"]["number"].update(span=[0, 3]),
        lambda r: r["slots"]["number"].update(text="7"),
        lambda r: r["slots"]["number"].update(text=""),
        lambda r: r["slots"]["number"].update(unexpected=True),
    ],
)
def test_malformed_annotations_fail_instead_of_successful_empty(synthetic_row, mutate):
    mutate(synthetic_row)
    with pytest.raises(ValueError):
        public.nlupp_to_ticket(synthetic_row, 0)


def test_read_source_rejects_symlink_oversize_and_hash_mismatch(tmp_path):
    source = tmp_path / "source.json"
    source.write_bytes(b"[]")
    digest = hashlib.sha256(b"[]").hexdigest()
    assert public.read_pinned_file(source, digest) == b"[]"
    link = tmp_path / "link.json"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="non-symlink"):
        public.read_pinned_file(link, digest)
    with pytest.raises(ValueError, match="SHA256"):
        public.read_pinned_file(source, "0" * 64)
    source.write_bytes(b"x" * (public.MAX_SOURCE_BYTES + 1))
    with pytest.raises(ValueError, match="byte budget"):
        public.read_pinned_file(source, digest)


@pytest.fixture
def synthetic_pinned_files(tmp_path, monkeypatch, synthetic_row):
    source = tmp_path / "upstream.json"
    license_file = tmp_path / "upstream-license.txt"
    source.write_text(json.dumps([synthetic_row] * 104), encoding="utf-8")
    license_file.write_text("synthetic license fixture", encoding="utf-8")
    monkeypatch.setattr(public, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(
        public, "LICENSE_SHA256", hashlib.sha256(license_file.read_bytes()).hexdigest()
    )
    return source, license_file


def test_offline_outputs_are_reproducible_and_rows_have_verifiable_hashes(
    tmp_path, synthetic_pinned_files, monkeypatch
):
    def no_network(*args, **kwargs):
        pytest.fail("offline path tried network")

    monkeypatch.setattr(public, "download_pinned", no_network)
    source, license_file = synthetic_pinned_files
    outputs = [tmp_path / "one", tmp_path / "two"]
    manifests = [public.prepare_benchmark(out, 10, source, license_file) for out in outputs]
    assert manifests[0] == manifests[1]
    for name in ["source.json", "LICENSE.txt", "tickets.jsonl", "manifest.json"]:
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()
    canonical = (outputs[0] / "tickets.jsonl").read_bytes()
    assert hashlib.sha256(canonical).hexdigest() == manifests[0]["canonical_sha256"]
    for row, record in zip(canonical.splitlines(), manifests[0]["records"], strict=True):
        assert hashlib.sha256(row).hexdigest() == record["canonical_record_sha256"]
    before = (outputs[0] / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        public.prepare_benchmark(outputs[0], 10, source, license_file)
    assert (outputs[0] / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("count", [0, 21, True])
def test_unbounded_or_invalid_count_is_rejected_before_io(tmp_path, count):
    with pytest.raises(ValueError, match="count"):
        public.prepare_benchmark(tmp_path / "out", count)


def test_tampered_offline_source_creates_no_success_artifact(tmp_path, synthetic_pinned_files):
    source, license_file = synthetic_pinned_files
    source.write_bytes(source.read_bytes() + b" ")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="SHA256"):
        public.prepare_benchmark(output, 10, source, license_file)
    assert not output.exists()


def test_duplicate_json_keys_are_rejected_even_with_verified_file_hash(
    tmp_path, synthetic_pinned_files, monkeypatch
):
    source, license_file = synthetic_pinned_files
    source.write_text('[{"text":"one","text":"two"}]', encoding="utf-8")
    monkeypatch.setattr(public, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="duplicate JSON key"):
        public.prepare_benchmark(tmp_path / "out", 1, source, license_file)


class FakeResponse(io.BytesIO):
    def geturl(self):
        return public.SOURCE_URL


def test_downloader_rejects_unpinned_url_and_bounds_network_read(monkeypatch):
    with pytest.raises(ValueError, match="unrecognised"):
        public.download_pinned("http://example.invalid/source", public.SOURCE_SHA256)
    response = FakeResponse(b"x" * (public.MAX_SOURCE_BYTES + 100))
    monkeypatch.setattr(public.urllib.request, "urlopen", lambda *args, **kwargs: response)
    with pytest.raises(ValueError, match="byte budget"):
        public.download_pinned(public.SOURCE_URL, public.SOURCE_SHA256)


def test_tls_fallback_preserves_certificate_and_hostname_verification(monkeypatch):
    contexts = []

    def fail_cert(url, *, timeout, context):
        contexts.append(context)
        raise urllib.error.URLError(ssl.SSLCertVerificationError("synthetic failure"))

    monkeypatch.setattr(public.urllib.request, "urlopen", fail_cert)
    with pytest.raises(urllib.error.URLError):
        public.download_pinned(public.SOURCE_URL, public.SOURCE_SHA256)
    assert contexts
    assert all(context.verify_mode == ssl.CERT_REQUIRED for context in contexts)
    assert all(context.check_hostname for context in contexts)


def test_invalid_index_and_partial_offline_configuration_are_rejected(tmp_path, synthetic_row):
    for index in [-1, True]:
        with pytest.raises(ValueError, match="index"):
            public.nlupp_to_ticket(copy.deepcopy(synthetic_row), index)
    with pytest.raises(ValueError, match="both source and license"):
        public.prepare_benchmark(tmp_path / "out", source_file=tmp_path / "source")
