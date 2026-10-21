"""Internal worker. All stdout is aggregate machine-readable metrics."""

import time

ENTRY = time.perf_counter()

import json  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import psutil  # noqa: E402

from compactfacts.data import read_tickets  # noqa: E402
from compactfacts.models import create_extractor  # noqa: E402
from compactfacts.schemas import InferenceConfig  # noqa: E402
from compactfacts.utils.benchmark import percentile  # noqa: E402
from compactfacts.utils.io import file_hash, strict_loads  # noqa: E402
from compactfacts.utils.provenance import provenance  # noqa: E402


def _weight_bytes(path):
    if path is None:
        return 0
    root = Path(path)
    return sum(
        p.stat().st_size
        for p in root.rglob("*")
        if p.is_file() and p.suffix in {".safetensors", ".bin", ".gguf", ".pt", ".pth"}
    )


def main():
    path, config_path, repeats, warmup = sys.argv[1:]
    tickets = read_tickets(path)
    if not tickets:
        raise ValueError("benchmark requires input")
    config = InferenceConfig.model_validate(strict_loads(Path(config_path).read_text()))
    load_start = time.perf_counter()
    extractor = create_extractor(config)
    prepare = getattr(extractor, "prepare", None)
    if prepare is not None:
        prepare()
    load_seconds = time.perf_counter() - load_start
    torch = sys.modules.get("torch")
    cuda = torch is not None and config.device == "cuda" and torch.cuda.is_available()
    mps = torch is not None and config.device == "mps" and torch.backends.mps.is_available()

    def sync():
        if cuda:
            torch.cuda.synchronize()
        elif mps:
            torch.mps.synchronize()

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    first = extractor.predict(tickets[0])
    sync()
    cold = time.perf_counter() - ENTRY
    for _ in range(int(warmup)):
        for ticket in tickets:
            extractor.predict(ticket)
    sync()
    timings, tokens = [], []
    successes = 0
    total_start = time.perf_counter()
    for _ in range(int(repeats)):
        for ticket in tickets:
            sync()
            start = time.perf_counter()
            prediction = extractor.predict(ticket)
            sync()
            timings.append(time.perf_counter() - start)
            successes += prediction.status == "ok"
            tokens.append(prediction.generated_tokens)
    elapsed = time.perf_counter() - total_start
    highwater = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    highwater_bytes = int(highwater if sys.platform == "darwin" else highwater * 1024)
    result = {
        "measurement_status": "measured" if successes == len(timings) else "failed_predictions",
        "backend": config.backend,
        "cold_startup_seconds": cold,
        "cold_startup_scope": (
            "worker module entry through imports, input read, model load and first prediction; "
            "excludes process spawn"
        ),
        "cold_first_prediction_status": first.status,
        "model_load_seconds": load_seconds,
        "warm_latency_p50_seconds": percentile(timings, 0.5),
        "warm_latency_p95_seconds": percentile(timings, 0.95),
        "warm_elapsed_seconds": elapsed,
        "samples": len(timings),
        "successes": successes,
        "samples_per_second": len(timings) / elapsed,
        "successful_samples_per_second": successes / elapsed,
        "peak_ram_bytes": highwater_bytes,
        "peak_ram_method": (
            "fresh worker process OS ru_maxrss high-water; includes interpreter and imports"
        ),
        "rss_after_bytes": psutil.Process().memory_info().rss,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
        "vram_note": "CUDA allocator peak excludes driver/untracked memory"
        if cuda
        else "CUDA backend not active; MPS peak is not substituted",
        "model_size_bytes": _weight_bytes(config.model_path),
        "adapter_size_bytes": _weight_bytes(config.adapter_path),
        "size_method": "logical serialized weight-file bytes; rules/empty have zero model weights",
        "tokens_per_second": sum(tokens) / elapsed if all(x is not None for x in tokens) else None,
        "token_note": "generated tokens, including retries"
        if all(x is not None for x in tokens)
        else "backend does not emit model tokens",
        "warmup_passes": int(warmup),
        "repeats": int(repeats),
        "batch_policy": "serial, batch_size=1",
        "percentile_method": "linear interpolation",
        "input_sha256": file_hash(path),
        "config": config.model_dump(),
        "environment": provenance(),
    }
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
