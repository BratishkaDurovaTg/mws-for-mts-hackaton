"""Capture actual HF outputs on explicitly synthetic fixtures for runtime diagnosis."""

import argparse
from pathlib import Path

from compactfacts.data import read_tickets
from compactfacts.evaluation import evaluate
from compactfacts.models.hf import HFExtractor
from compactfacts.models.prompts import build_messages, target_json
from compactfacts.schemas import InferenceConfig
from compactfacts.utils.io import file_hash, write_json
from compactfacts.utils.provenance import provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("use a new output file")
    tickets = read_tickets(args.input)
    if not tickets or any(t.metadata.get("synthetic") is not True for t in tickets):
        parser.error("raw-output capture requires explicitly synthetic fixtures")
    config = InferenceConfig.model_validate_json(Path(args.config).read_text())
    extractor = HFExtractor(config)
    if not extractor.prepare():
        raise ValueError("real local model could not load")
    generated = []
    actual_generate = extractor._model.generate

    def record_generation(*positional, **kwargs):
        # Always execute the real pretrained model; only retain its actual output IDs.
        result = actual_generate(*positional, **kwargs)
        generated.append(result[0].tolist()[kwargs["input_ids"].shape[1] :])
        return result

    extractor._model.generate = record_generation
    predictions, examples = [], []
    for ticket in tickets:
        generated.clear()
        prediction = extractor.predict(ticket)
        predictions.append(prediction)
        examples.append(
            {
                "input": ticket.model_dump(mode="json"),
                "prompt": build_messages(ticket, config.allowed_types),
                "gold_target": target_json(ticket),
                "raw_outputs": [
                    extractor._tokenizer.decode(ids, skip_special_tokens=True) for ids in generated
                ],
                "prediction": prediction.model_dump(mode="json"),
            }
        )
    torch = extractor._torch
    mps = config.device == "mps"
    write_json(
        args.output,
        {
            **provenance(),
            "scope": "real pretrained synthetic-only diagnostic; NOT MWS QUALITY",
            "input_sha256": file_hash(args.input),
            "config": config.model_dump(),
            "examples": examples,
            "metrics": evaluate(tickets, predictions),
            "mps_current_allocated_bytes": torch.mps.current_allocated_memory() if mps else None,
            "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory() if mps else None,
            "mps_memory_scope": "end-of-run snapshots, not peaks",
        },
    )
    print(f"captured {len(examples)} real synthetic-only generations")


if __name__ == "__main__":
    main()
