"""Replay the small core pipeline; optionally mirror explicitly synthetic results for review."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from compactfacts.utils.io import read_jsonl, write_json
from compactfacts.utils.ledger import append_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/preparation-replay"))
    parser.add_argument("--id-prefix", default="preparation-20261022")
    parser.add_argument("--publish-synthetic", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output directory")
    source = "tests/fixtures/synthetic.jsonl"
    if not all(row.get("metadata", {}).get("synthetic") is True for row in read_jsonl(source)):
        parser.error("preparation replay only accepts explicitly synthetic fixtures")

    def cli(*command):
        result = subprocess.run(
            [sys.executable, "-m", "compactfacts", *map(str, command)],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    cli("validate", "--input", source)
    cli("schema", "--output", args.output / "schema.json")
    cli("eda", "--input", source, "--output", args.output / "eda.json")
    cli("split", "--input", source, "--output", args.output / "split")
    for backend in ["empty", "rules"]:
        cli(
            "run",
            "--input",
            source,
            "--config",
            f"configs/{backend}.json",
            "--output",
            args.output / backend,
            "--id",
            f"{args.id_prefix}-{backend}",
            "--ledger",
            args.output / "runs.jsonl",
            "--repeats",
            "10",
            "--warmup",
            "1",
            "--span-diagnostic",
            "--hypothesis",
            "Exact-mention software contracts remain reproducible on frozen fixtures",
            "--change",
            "Empty predictor" if backend == "empty" else "Empty predictor to exact rules",
            "--expected-result",
            "Zero F1" if backend == "empty" else "Frozen fixture F1 0.590909",
            "--decision-rule",
            "KEEP only on zero technical failures and expected frozen fixture counts",
            "--conclusion",
            "Synthetic-only software reference; no model-quality improvement claim",
            "--decision",
            "KEEP",
            "--reason",
            "Retain a measured regression reference",
        )
    cli(
        "submission",
        "--predictions",
        args.output / "rules/predictions.jsonl",
        "--output",
        args.output / "canonical-export.jsonl",
    )
    cli(
        "compare",
        "--runs",
        args.output / "empty/run.json",
        args.output / "rules/run.json",
        "--output",
        args.output / "comparison.json",
    )
    records = read_jsonl(args.output / "runs.jsonl")
    for record in records:
        expected = 0 if record["config"]["backend"] == "empty" else 13 / 22
        if (
            record["metrics"]["technical_failure_rate"]
            or abs(record["metrics"]["micro"]["f1"] - expected) > 1e-12
        ):
            raise ValueError("frozen software baseline acceptance failed")
    if args.publish_synthetic:
        target = Path("experiments/baselines")
        if target.exists():
            raise ValueError("published snapshot already exists; preserve history")
        target.mkdir(parents=True)
        for filename in ["eda.json", "comparison.json"]:
            shutil.copyfile(args.output / filename, target / filename)
        write_json(
            target / "split_manifest.json",
            json.loads((args.output / "split/manifest.json").read_text()),
        )
        for record in records:
            backend = record["config"]["backend"]
            shutil.copytree(args.output / backend, target / backend)
            record["published_artifact"] = str(target / backend)
            append_run("experiments/runs.jsonl", record)
    print(json.dumps({"status": "ok", "runs": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
