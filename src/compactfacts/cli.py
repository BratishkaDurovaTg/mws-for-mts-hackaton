"""Small reproducible CLI. Default logs contain counts/codes, never ticket text."""

import argparse
import importlib
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from compactfacts.data import (
    MappingAdapter,
    analyze,
    export_submission,
    read_tickets,
    split_tickets,
    write_tickets,
)
from compactfacts.evaluation import evaluate
from compactfacts.inference import predict_batch
from compactfacts.models import create_extractor
from compactfacts.schemas import InferenceConfig, Prediction, Ticket
from compactfacts.utils.benchmark import benchmark
from compactfacts.utils.io import (
    file_hash,
    read_jsonl,
    stable_hash,
    strict_loads,
    write_json,
    write_jsonl,
)
from compactfacts.utils.ledger import append_run
from compactfacts.utils.provenance import provenance


def _json_file(path):
    return strict_loads(Path(path).read_text(encoding="utf-8"))


def _config(path):
    return InferenceConfig.model_validate(_json_file(path))


def _predictions(path):
    return [Prediction.model_validate(row) for row in read_jsonl(path)]


def _plugin(spec):
    if spec is None:
        return None
    module, attr = spec.split(":", 1)
    value = getattr(importlib.import_module(module), attr)
    if not callable(value):
        raise ValueError("plugin must be callable")
    return value


def _check_output(path, inputs=(), force=False):
    output = Path(path).resolve()
    if any(output == Path(x).resolve() for x in inputs if x):
        raise ValueError("output cannot overwrite input")
    if any(output in Path(x).resolve().parents for x in inputs if x):
        raise ValueError("output directory cannot contain an input")
    if output.exists() and not force:
        raise ValueError("output exists; use a new path or --force")


def parser():
    result = argparse.ArgumentParser(description="Provisional MWS fact extraction preparation")
    result.add_argument(
        "--debug", action="store_true", help="safe diagnostics: exception class only"
    )
    commands = result.add_subparsers(dest="command", required=True)
    for name in ["validate", "eda", "split", "adapt", "infer", "benchmark", "run"]:
        cmd = commands.add_parser(name)
        cmd.add_argument("--input", required=True)
        if name != "validate":
            cmd.add_argument("--output", required=True)
            cmd.add_argument("--force", action="store_true")
        if name in {"infer", "benchmark", "run"}:
            cmd.add_argument("--config", default="configs/rules.json")
        if name in {"benchmark", "run"}:
            cmd.add_argument("--repeats", type=int, default=10)
            cmd.add_argument("--warmup", type=int, default=1)
        if name == "adapt":
            cmd.add_argument(
                "--mapping", required=True, help="JSON MappingAdapter constructor fields"
            )
        if name == "split":
            cmd.add_argument("--dev-fraction", type=float, default=0.25)
            cmd.add_argument("--seed", type=int, default=42)
            cmd.add_argument("--near-threshold", type=float, default=0.9)
        if name == "run":
            cmd.add_argument("--id", required=True)
            cmd.add_argument("--ledger", default="experiments/runs.jsonl")
            cmd.add_argument("--hypothesis", required=True)
            cmd.add_argument("--change", required=True)
            cmd.add_argument("--expected-result", required=True)
            cmd.add_argument("--decision-rule", required=True)
            cmd.add_argument("--conclusion", required=True)
            cmd.add_argument("--decision", choices=["KEEP", "REJECT"], required=True)
            cmd.add_argument("--reason", required=True)
            cmd.add_argument("--scorer", help="reviewed local module:callable, official primary")
            cmd.add_argument("--span-diagnostic", action="store_true")
    cmd = commands.add_parser("evaluate")
    cmd.add_argument("--gold", required=True)
    cmd.add_argument("--predictions", required=True)
    cmd.add_argument("--output", required=True)
    cmd.add_argument("--normalization", help="explicit per-type normalization JSON")
    cmd.add_argument("--scorer", help="reviewed local module:callable")
    cmd.add_argument("--span-diagnostic", action="store_true")
    cmd.add_argument("--force", action="store_true")
    cmd = commands.add_parser("submission")
    cmd.add_argument("--predictions", required=True)
    cmd.add_argument("--output", required=True)
    cmd.add_argument("--official", action="store_true")
    cmd.add_argument("--adapter", help="reviewed local module:adapter_factory")
    cmd.add_argument("--force", action="store_true")
    cmd = commands.add_parser("schema")
    cmd.add_argument("--output", required=True)
    cmd.add_argument("--force", action="store_true")
    cmd = commands.add_parser("compare")
    cmd.add_argument("--runs", nargs="+", required=True)
    cmd.add_argument("--output", required=True)
    cmd.add_argument("--force", action="store_true")
    cmd = commands.add_parser("train")
    cmd.add_argument("--config", required=True)
    cmd.add_argument("--train", required=True)
    cmd.add_argument("--dev", required=True)
    cmd.add_argument("--id", required=True)
    cmd.add_argument("--hypothesis", required=True)
    cmd.add_argument("--change", required=True)
    cmd.add_argument("--expected-result", required=True)
    cmd.add_argument("--decision-rule", required=True)
    cmd.add_argument("--ledger", default="experiments/runs.jsonl")
    cmd.add_argument("--scorer", help="reviewed local module:callable, official primary")
    return result


def execute(args):
    command = args.command
    if command in {"run", "train"} and any(
        not getattr(args, key).strip()
        for key in ["hypothesis", "change", "expected_result", "decision_rule"]
    ):
        raise ValueError("experiment contract must be nonempty before computation")
    if hasattr(args, "output"):
        _check_output(
            args.output,
            [
                getattr(args, x, None)
                for x in ["input", "gold", "predictions", "config", "mapping", "normalization"]
            ],
            args.force,
        )
    if command == "schema":
        write_json(
            args.output,
            {
                "status": "provisional",
                "ticket": Ticket.model_json_schema(),
                "prediction": Prediction.model_json_schema(),
            },
        )
        return {"status": "ok"}
    if command == "compare":
        from compactfacts.evaluation.comparison import compare_runs

        _check_output(args.output, args.runs, args.force)
        write_json(args.output, compare_runs([_json_file(path) for path in args.runs]))
        return {"status": "ok", "runs": len(args.runs)}
    if command == "train":
        from compactfacts.training.runner import TrainingConfig, run_training

        config = TrainingConfig.model_validate(_json_file(args.config))
        inputs = [
            args.train,
            args.dev,
            args.config,
            config.split_manifest_path,
            config.prior_gate_manifest,
            getattr(config, "baseline_run_path", None),
        ]
        _check_output(config.output_dir, inputs, force=True)
        _check_output(args.ledger, inputs, force=True)
        if Path(config.output_dir).resolve() in Path(args.ledger).resolve().parents:
            raise ValueError("training ledger must be outside the output artifact directory")
        if Path(args.ledger).exists() and any(
            row.get("id") == args.id for row in read_jsonl(args.ledger)
        ):
            raise ValueError("experiment id already exists")
        train_tickets, dev_tickets = read_tickets(args.train), read_tickets(args.dev)
        manifest_path = Path(config.output_dir) / "training_manifest.json"
        try:
            manifest = run_training(config, train_tickets, dev_tickets, _plugin(args.scorer))
        except Exception:
            if not manifest_path.is_file():
                raise
            manifest = _json_file(manifest_path)
        passed = manifest.get("status") == "passed"
        append_run(
            args.ledger,
            {
                "id": args.id,
                **provenance(),
                "hypothesis": args.hypothesis,
                "change": args.change,
                "expected_result": args.expected_result,
                "decision_rule": args.decision_rule,
                "base_model": config.inference.model_id,
                "revision": config.inference.revision,
                "data_sha256": {"train": file_hash(args.train), "dev": file_hash(args.dev)},
                "config": config.model_dump(),
                "command": ["compactfacts", *sys.argv[1:]],
                "seed": config.inference.seed,
                "metrics": manifest.get("validation_metrics"),
                "benchmark": {
                    "duration_seconds": manifest.get("duration_seconds"),
                    "cuda_peak_allocated_bytes": manifest.get("cuda_peak_allocated_bytes"),
                    "inference_latency": None,
                },
                "artifact": str(manifest_path),
                "decision": "KEEP" if passed else "REJECT",
                "conclusion": "stage gate passed; quality improvement not established"
                if passed
                else "stage gate failed; inspect safe manifest",
                "reason": "measured stage gate only; compare against frozen baseline separately",
            },
        )
        return {
            "status": "ok" if passed else "partial_failure",
            "stage": config.stage,
            "gate": manifest.get("status"),
            "id": args.id,
        }
    if command == "evaluate":
        metrics = evaluate(
            read_tickets(args.gold),
            _predictions(args.predictions),
            normalization=_json_file(args.normalization) if args.normalization else None,
            official_scorer=_plugin(args.scorer),
        )
        if args.span_diagnostic:
            from compactfacts.evaluation.spans import evaluate_spans

            metrics["span_diagnostic"] = evaluate_spans(
                read_tickets(args.gold), _predictions(args.predictions)
            )
        write_json(args.output, metrics)
        return {"status": "ok", "primary": metrics["primary"]}
    if command == "submission":
        factory = _plugin(args.adapter)
        export_submission(
            args.output,
            _predictions(args.predictions),
            adapter=factory() if factory else None,
            official=args.official,
        )
        return {
            "status": "ok",
            "format": "official_adapter" if args.official else "canonical_diagnostic",
        }
    if command == "benchmark":
        report = benchmark(args.input, args.config, args.repeats, args.warmup)
        write_json(args.output, report)
        return {
            "status": "ok" if report["measurement_status"] == "measured" else "partial_failure",
            "warm_latency_p95_seconds": report["warm_latency_p95_seconds"],
        }
    adapter = MappingAdapter(**_json_file(args.mapping)) if command == "adapt" else None
    tickets = read_tickets(args.input, adapter)
    if command == "validate":
        return {
            "status": "ok",
            "records": len(tickets),
            "labelled": sum(t.facts is not None for t in tickets),
            "input_sha256": file_hash(args.input),
        }
    if command == "adapt":
        write_tickets(args.output, tickets)
    elif command == "eda":
        write_json(args.output, analyze(tickets))
    elif command == "split":
        train, dev, manifest = split_tickets(
            tickets, args.dev_fraction, args.seed, args.near_threshold
        )
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=args.force)
        write_tickets(output / "train.jsonl", train)
        write_tickets(output / "dev.jsonl", dev)
        write_json(output / "manifest.json", manifest)
    elif command == "infer":
        predictions = predict_batch(create_extractor(_config(args.config)), tickets)
        write_jsonl(args.output, [p.model_dump(mode="json") for p in predictions])
        return {
            "status": "ok" if all(p.status == "ok" for p in predictions) else "partial_failure",
            "records": len(predictions),
            "errors": sum(p.status != "ok" for p in predictions),
        }
    elif command == "run":
        ledger = Path(args.ledger).resolve()
        _check_output(ledger, [args.input, args.config], force=True)
        children = [
            Path(args.output).resolve() / name
            for name in ["predictions.jsonl", "metrics.json", "benchmark.json", "run.json"]
        ]
        if ledger in children:
            raise ValueError("ledger cannot share an artifact path")
        if ledger.exists() and any(row.get("id") == args.id for row in read_jsonl(ledger)):
            raise ValueError("experiment id already exists")
        config = _config(args.config)
        if any(t.facts is None for t in tickets) or not tickets:
            raise ValueError("run requires nonempty labelled evaluation data")
        if args.span_diagnostic:
            from compactfacts.evaluation.spans import evaluate_spans

            evaluate_spans(tickets, [])  # Validate precise gold spans before model computation.
        report = benchmark(args.input, args.config, args.repeats, args.warmup)
        extractor = create_extractor(config)
        predictions = predict_batch(extractor, tickets)
        metrics = evaluate(tickets, predictions, official_scorer=_plugin(args.scorer))
        if args.span_diagnostic:
            from compactfacts.evaluation.spans import evaluate_spans

            metrics["span_diagnostic"] = evaluate_spans(tickets, predictions)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=args.force)
        write_jsonl(output / "predictions.jsonl", [p.model_dump(mode="json") for p in predictions])
        write_json(output / "metrics.json", metrics)
        write_json(output / "benchmark.json", report)
        record = {
            "id": args.id,
            **provenance(),
            "hypothesis": args.hypothesis,
            "change": args.change,
            "expected_result": args.expected_result,
            "decision_rule": args.decision_rule,
            "base_model": config.model_id,
            "revision": config.revision,
            "model_weight_sha256": {
                p.name: file_hash(p) for p in sorted(Path(config.model_path).glob("*.safetensors"))
            }
            if config.backend == "hf"
            else {},
            "data_sha256": file_hash(args.input),
            "canonical_data_sha256": stable_hash(
                [t.model_dump(mode="json") for t in sorted(tickets, key=lambda t: t.id)]
            ),
            "config": config.model_dump(),
            "command": ["compactfacts", *sys.argv[1:]],
            "seed": config.seed,
            "metrics": metrics,
            "benchmark": report,
            "artifact": str(output),
            "conclusion": args.conclusion,
            "decision": args.decision,
            "reason": args.reason,
            "synthetic_only": all(t.metadata.get("synthetic") is True for t in tickets),
        }
        write_json(output / "run.json", record)
        append_run(args.ledger, record)
        return {
            "status": "ok" if metrics["technical_failure_rate"] == 0 else "partial_failure",
            "id": args.id,
            "primary": metrics["primary"],
        }
    return {"status": "ok", "records": len(tickets)}


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = execute(args)
    except Exception as exc:
        # ValidationError includes raw input by default. Do not expose it in logs.
        code = (
            "invalid_input_or_config"
            if isinstance(exc, (ValueError, ValidationError))
            else "operation_failed"
        )
        message = {"status": "error", "error_code": code}
        if args.debug:
            message["exception_class"] = type(exc).__name__
        print(json.dumps(message), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
