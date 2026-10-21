"""Adversarial regressions found during independent boundary/provenance review."""

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from compactfacts.cli import main
from compactfacts.data import write_tickets
from compactfacts.evaluation import evaluate
from compactfacts.schemas import InferenceConfig, Message, Prediction, Ticket
from compactfacts.utils.ledger import append_run


def _ticket(identifier="a", text="Example support request."):
    return Ticket(
        id=identifier,
        messages=[Message(id="m", role="user", content=text)],
        facts=[],
    )


def _ledger_record(identifier):
    return {
        "id": identifier,
        "timestamp_utc": "2026-10-21T00:00:00Z",
        "git_commit": "synthetic-test-commit",
        "hypothesis": "fixture only",
        "data_sha256": "fixture",
        "config": {},
        "command": [],
        "seed": 42,
        "metrics": {},
        "benchmark": {},
        "artifact": "fixture",
        "conclusion": "software contract only",
        "decision": "KEEP",
        "reason": "test",
    }


def test_failed_retries_remain_visible_in_rates():
    failed = Prediction(
        ticket_id="a", status="error", facts=[], error_code="invalid_json", attempts=3
    )
    result = evaluate([_ticket()], [failed])
    assert result["retried_rate"] == 1.0
    assert result["parse_failure_rate"] == 1.0
    assert result["success_rate"] == 0.0
    assert result["exact_fact_set_accuracy"] == 0.0


@pytest.mark.parametrize("field", ["name", "provenance"])
def test_official_primary_requires_nonblank_attribution(field):
    official = {"name": "fixture", "provenance": "local fixture", "score": 0.25}
    official[field] = " \t"
    with pytest.raises(ValueError, match="official scorer"):
        evaluate(
            [_ticket()],
            [Prediction(ticket_id="a", status="ok", facts=[])],
            official_scorer=lambda gold, predictions: official,
        )


def test_ledger_without_final_newline_is_refused_without_mutation(tmp_path):
    path = tmp_path / "runs.jsonl"
    original = json.dumps(_ledger_record("first")).encode()
    path.write_bytes(original)
    with pytest.raises(ValueError, match="final line"):
        append_run(path, _ledger_record("second"))
    assert path.read_bytes() == original
    assert json.loads(path.read_text())["id"] == "first"


def test_split_cannot_overwrite_raw_input_via_output_directory(tmp_path, capsys):
    source = tmp_path / "train.jsonl"
    write_tickets(
        source,
        [
            _ticket("a", "Certificate renewal needs approval from the security team."),
            _ticket("b", "Нужен отдельный счёт на оплату подписки за следующий год."),
        ],
    )
    original = source.read_bytes()
    assert main(["split", "--input", str(source), "--output", str(tmp_path), "--force"]) == 2
    assert source.read_bytes() == original
    assert not (tmp_path / "dev.jsonl").exists()
    assert json.loads(capsys.readouterr().err)["status"] == "error"


def test_evaluate_output_cannot_overwrite_normalization_config(tmp_path, capsys):
    gold = tmp_path / "gold.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    normalization = tmp_path / "normalization.json"
    write_tickets(gold, [_ticket()])
    predictions.write_text(
        Prediction(ticket_id="a", status="ok", facts=[]).model_dump_json() + "\n"
    )
    normalization.write_text('{"service": "casefold"}\n')
    original = normalization.read_bytes()
    assert (
        main(
            [
                "evaluate",
                "--gold",
                str(gold),
                "--predictions",
                str(predictions),
                "--normalization",
                str(normalization),
                "--output",
                str(normalization),
                "--force",
            ]
        )
        == 2
    )
    assert normalization.read_bytes() == original
    assert json.loads(capsys.readouterr().err)["status"] == "error"


def test_run_ledger_cannot_append_into_raw_dataset(tmp_path, capsys):
    source = tmp_path / "gold.jsonl"
    config = tmp_path / "config.json"
    output = tmp_path / "artifacts"
    write_tickets(source, [_ticket()])
    config.write_text(InferenceConfig(backend="empty").model_dump_json())
    original = source.read_bytes()
    assert (
        main(
            [
                "run",
                "--input",
                str(source),
                "--config",
                str(config),
                "--output",
                str(output),
                "--ledger",
                str(source),
                "--id",
                "fixture",
                "--hypothesis",
                "test",
                "--conclusion",
                "test",
                "--decision",
                "KEEP",
                "--reason",
                "test",
            ]
        )
        == 2
    )
    assert source.read_bytes() == original
    assert not output.exists()
    assert json.loads(capsys.readouterr().err)["status"] == "error"


def test_duplicate_run_id_cannot_replace_previous_artifacts(tmp_path, monkeypatch, capsys):
    cli = importlib.import_module("compactfacts.cli")
    source, config = tmp_path / "gold.jsonl", tmp_path / "config.json"
    output, ledger = tmp_path / "artifacts", tmp_path / "runs.jsonl"
    write_tickets(source, [_ticket()])
    config.write_text(InferenceConfig(backend="empty").model_dump_json())
    append_run(ledger, _ledger_record("existing"))
    output.mkdir()
    artifact = output / "run.json"
    artifact.write_text('{"original": true}\n')
    original_artifact, original_ledger = artifact.read_bytes(), ledger.read_bytes()

    def forbidden_benchmark(*args, **kwargs):
        pytest.fail("duplicate run must fail before expensive measurement or artifact writes")

    monkeypatch.setattr(cli, "benchmark", forbidden_benchmark)
    assert (
        main(
            [
                "run",
                "--input",
                str(source),
                "--config",
                str(config),
                "--output",
                str(output),
                "--ledger",
                str(ledger),
                "--id",
                "existing",
                "--hypothesis",
                "test",
                "--conclusion",
                "changed",
                "--decision",
                "KEEP",
                "--reason",
                "test",
                "--force",
            ]
        )
        == 2
    )
    assert artifact.read_bytes() == original_artifact
    assert ledger.read_bytes() == original_ledger
    assert json.loads(capsys.readouterr().err)["status"] == "error"


def test_provenance_uses_invocation_checkout_and_distinguishes_unknown_git(monkeypatch, tmp_path):
    provenance = importlib.import_module("compactfacts.utils.provenance")
    calls = []
    original_run = subprocess.run

    def no_git(command, **kwargs):
        if command[0] != "git":
            return original_run(command, **kwargs)
        calls.append(kwargs["cwd"])
        return subprocess.CompletedProcess(command, 128, "", "")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(provenance.subprocess, "run", no_git)
    result = provenance.provenance()
    assert all(Path(cwd) == tmp_path for cwd in calls)
    assert result["git_commit"] is None
    assert result["git_dirty"] is None
    assert len(result["package_source_sha256"]) == 64


def test_lazy_model_preparation_is_timed_before_cuda_probe(tmp_path, monkeypatch, capsys):
    worker = importlib.import_module("compactfacts._benchmark_worker")
    clock = {"now": 100.0}
    state = {"prepared": False, "syncs": 0, "reset": False}

    def perf_counter():
        clock["now"] += 0.001
        return clock["now"]

    def synchronize():
        state["syncs"] += 1
        clock["now"] += 0.002

    def reset_peak():
        assert state["prepared"]
        state["reset"] = True

    cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=synchronize,
        reset_peak_memory_stats=reset_peak,
        max_memory_allocated=lambda: 1234,
        max_memory_reserved=lambda: 5678,
    )
    fake_torch = SimpleNamespace(
        cuda=cuda,
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )

    class LazyExtractor:
        def prepare(self):
            clock["now"] += 0.05
            state["prepared"] = True
            monkeypatch.setitem(sys.modules, "torch", fake_torch)

        def predict(self, ticket):
            assert state["prepared"]
            clock["now"] += 0.005
            return Prediction(ticket_id=ticket.id, status="ok", facts=[], generated_tokens=2)

    config = tmp_path / "config.json"
    config.write_text(
        InferenceConfig(
            backend="hf",
            model_path="synthetic-local-fixture",
            model_id="synthetic-fixture",
            revision="a" * 40,
            device="cuda",
        ).model_dump_json()
    )
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(worker, "time", SimpleNamespace(perf_counter=perf_counter))
    monkeypatch.setattr(worker, "ENTRY", 0.0)
    monkeypatch.setattr(worker, "read_tickets", lambda path: [_ticket()])
    monkeypatch.setattr(worker, "create_extractor", lambda config: LazyExtractor())
    monkeypatch.setattr(worker, "provenance", lambda: {})
    monkeypatch.setattr(worker, "file_hash", lambda path: "synthetic-fixture")
    monkeypatch.setattr(worker, "_weight_bytes", lambda path: 0)
    monkeypatch.setattr(sys, "argv", ["worker", "fixture.jsonl", str(config), "2", "0"])
    worker.main()
    report = json.loads(capsys.readouterr().out)
    assert report["model_load_seconds"] >= 0.05
    assert report["peak_vram_allocated_bytes"] == 1234
    assert report["peak_vram_reserved_bytes"] == 5678
    assert report["successes"] == report["samples"] == 2
    assert state["reset"] is True
    assert state["syncs"] >= 4
    assert report["warm_latency_p95_seconds"] > 0
