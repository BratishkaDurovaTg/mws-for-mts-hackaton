"""Compare recorded runs only when their evaluation protocols are compatible."""

import math


def compare_runs(runs: list[dict]) -> dict:
    if len(runs) < 2:
        raise ValueError("comparison requires at least two measured runs")

    def protocol(run):
        config, bench, metrics = run["config"], run["benchmark"], run["metrics"]
        env = bench["environment"]
        return (
            run["canonical_data_sha256"],
            metrics["primary"]["name"],
            metrics["primary"].get("provenance"),
            metrics["primary"].get("higher_is_better"),
            metrics["matching"],
            env["platform"],
            env["machine"],
            env["versions"],
            bench["batch_policy"],
            {
                k: config.get(k)
                for k in (
                    "seed",
                    "device",
                    "max_input_tokens",
                    "max_new_tokens",
                    "allowed_types",
                    "parser_mode",
                    "max_retries",
                    "require_evidence",
                    "enable_thinking",
                )
            },
        )

    reference = protocol(runs[0])
    primary = runs[0]["metrics"]["primary"]
    higher_is_better = primary.get(
        "higher_is_better", True if primary.get("status") != "official_plugin" else None
    )
    if type(higher_is_better) is not bool:
        raise ValueError("official score direction must be explicit before Pareto comparison")
    if any(protocol(run) != reference for run in runs[1:]):
        raise ValueError("runs differ in data/scorer/hardware/batch/decoding protocol")
    rows = []
    for run in runs:
        bench, metrics = run["benchmark"], run["metrics"]
        if bench["measurement_status"] != "measured" or metrics["technical_failure_rate"]:
            raise ValueError("failed predictions cannot enter a quality/resource ranking")
        row = {
            "id": run["id"],
            "score": metrics["primary"]["score"],
            "latency_p95_seconds": bench["warm_latency_p95_seconds"],
            "peak_ram_bytes": bench["peak_ram_bytes"],
            "weight_bytes": bench["model_size_bytes"] + bench["adapter_size_bytes"],
        }
        if any(
            type(value) not in {int, float}
            or not math.isfinite(value)
            or (key != "score" and value < 0)
            for key, value in row.items()
            if key != "id"
        ):
            raise ValueError("comparison metrics must be finite; resources must be nonnegative")
        rows.append(row)

    def dominates(a, b):
        sign = -1 if higher_is_better else 1
        left = (sign * a["score"], a["latency_p95_seconds"], a["peak_ram_bytes"], a["weight_bytes"])
        right = (
            sign * b["score"],
            b["latency_p95_seconds"],
            b["peak_ram_bytes"],
            b["weight_bytes"],
        )
        return all(x <= y for x, y in zip(left, right, strict=True)) and left != right

    for row in rows:
        row["pareto"] = not any(dominates(other, row) for other in rows)
    return {
        "status": "provisional"
        if runs[0]["metrics"]["metric_status"] == "provisional"
        and runs[0]["metrics"]["primary"].get("status") != "official_plugin"
        else "official_primary",
        "synthetic_only": all(run.get("synthetic_only") is True for run in runs),
        "higher_is_better": higher_is_better,
        "objectives": "explicit primary direction; minimize p95, process peak RAM, weight bytes",
        "rows": rows,
        "winner": None,
        "limitation": "Measured Pareto set for this protocol; no universal compactness cutoff.",
    }
