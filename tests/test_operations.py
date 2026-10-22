import json
from pathlib import Path

import pytest

from compactfacts.cli import main
from compactfacts.utils.benchmark import benchmark, percentile
from compactfacts.utils.io import read_jsonl, strict_loads, write_jsonl
from compactfacts.utils.ledger import append_run
from compactfacts.utils.provenance import provenance

FIXTURE = Path("tests/fixtures/synthetic.jsonl")


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}', '{"a":Infinity}'])
def test_strict_json_rejects_ambiguous_or_nonfinite(text):
    with pytest.raises(ValueError):
        strict_loads(text)


def test_jsonl_bounds_and_no_raw_error(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"secret":"credential",,}\n')
    with pytest.raises(ValueError, match="invalid JSONL") as error:
        read_jsonl(path)
    assert "credential" not in str(error.value)
    with pytest.raises(ValueError, match="budget"):
        read_jsonl(path, max_bytes=1)


def test_cli_end_to_end_and_no_silent_overwrite(tmp_path, capsys):
    prediction = tmp_path / "predictions.jsonl"
    metrics = tmp_path / "metrics.json"
    assert main(["validate", "--input", str(FIXTURE)]) == 0
    assert main(["infer", "--input", str(FIXTURE), "--output", str(prediction)]) == 0
    assert len(read_jsonl(prediction)) == 20
    assert (
        main(
            [
                "evaluate",
                "--gold",
                str(FIXTURE),
                "--predictions",
                str(prediction),
                "--output",
                str(metrics),
            ]
        )
        == 0
    )
    result = json.loads(metrics.read_text())
    assert result["records"] == 20
    assert 0 < result["micro"]["f1"] < 1
    assert main(["infer", "--input", str(FIXTURE), "--output", str(prediction)]) == 2
    assert "credential" not in capsys.readouterr().err
    assert (
        main(
            [
                "submission",
                "--predictions",
                str(prediction),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--official",
            ]
        )
        == 2
    )


def test_input_can_never_be_overwritten_even_with_force(tmp_path):
    path = tmp_path / "x.jsonl"
    write_jsonl(path, [])
    assert main(["infer", "--input", str(path), "--output", str(path), "--force"]) == 2
    assert path.read_text() == ""


def test_fresh_worker_profiles_without_fabricating_model_tokens():
    result = benchmark(FIXTURE, "configs/rules.json", repeats=2, warmup=0)
    assert result["samples"] == 40
    assert result["successes"] == 40
    assert 0 < result["warm_latency_p50_seconds"] <= result["warm_latency_p95_seconds"]
    assert result["cold_startup_seconds"] > 0
    assert result["peak_ram_bytes"] > 0
    assert result["tokens_per_second"] is None
    assert result["peak_vram_allocated_bytes"] is None
    assert result["model_size_bytes"] == 0


def test_ledger_duplicate_id_does_not_rewrite_history(tmp_path):
    record = dict.fromkeys(
        [
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
            "reason",
        ],
        None,
    )
    record.update(
        id="x",
        decision="KEEP",
        hypothesis="fixture",
        change="fixture",
        expected_result="fixture",
        decision_rule="fixture",
    )
    ledger = tmp_path / "runs.jsonl"
    append_run(ledger, record)
    before = ledger.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        append_run(ledger, record)
    assert ledger.read_bytes() == before


def test_provenance_has_source_fingerprint_without_customer_text():
    result = provenance()
    assert len(result["package_source_sha256"]) == 64
    assert result["versions"]["pydantic"] == "2.14.1"


def test_percentiles_and_budget_validation():
    assert percentile([1, 3], 0.5) == 2
    assert percentile([1, 3], 0.95) == 2.9
    with pytest.raises(ValueError):
        benchmark(FIXTURE, "configs/rules.json", repeats=0)


@pytest.mark.parametrize("field", ["hypothesis", "change", "expected_result", "decision_rule"])
@pytest.mark.parametrize("invalid", [None, " "])
def test_ledger_experiment_contract_rejects_missing_decision_before_write(tmp_path, field, invalid):
    from compactfacts.utils.ledger import append_run

    keys = [
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
        "reason",
    ]
    record = dict.fromkeys(keys, "fixture")
    record.update(id="x", decision="KEEP")
    record[field] = invalid
    path = tmp_path / "runs.jsonl"
    with pytest.raises(ValueError, match="four nonempty"):
        append_run(path, record)
    assert not path.exists()


def test_blank_decision_rule_prevents_benchmark_computation(tmp_path, monkeypatch):
    import compactfacts.cli as cli

    def forbidden(*args, **kwargs):
        pytest.fail("benchmark must not run without an explicit decision rule")

    monkeypatch.setattr(cli, "benchmark", forbidden)
    assert (
        main(
            [
                "run",
                "--input",
                str(FIXTURE),
                "--output",
                str(tmp_path / "out"),
                "--id",
                "x",
                "--hypothesis",
                "fixture",
                "--change",
                "fixture",
                "--expected-result",
                "fixture",
                "--decision-rule",
                " ",
                "--conclusion",
                "fixture",
                "--decision",
                "KEEP",
                "--reason",
                "fixture",
            ]
        )
        == 2
    )


def test_span_gold_is_validated_before_benchmark_computation(tmp_path, monkeypatch):
    import compactfacts.cli as cli

    def forbidden(*args, **kwargs):
        pytest.fail("contextual fact evidence is not exact entity-span gold")

    monkeypatch.setattr(cli, "benchmark", forbidden)
    assert (
        main(
            [
                "run",
                "--input",
                str(FIXTURE),
                "--output",
                str(tmp_path / "out"),
                "--id",
                "x",
                "--hypothesis",
                "fixture",
                "--change",
                "fixture",
                "--expected-result",
                "fixture",
                "--decision-rule",
                "fixture",
                "--conclusion",
                "fixture",
                "--decision",
                "KEEP",
                "--reason",
                "fixture",
                "--span-diagnostic",
            ]
        )
        == 2
    )
