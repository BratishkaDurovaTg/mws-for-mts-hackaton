from copy import deepcopy

import pytest

from compactfacts.evaluation.comparison import compare_runs


def measured(run_id, score, latency, ram=100, weights=100):
    return {
        "id": run_id,
        "canonical_data_sha256": "same-frozen-data",
        "config": {},
        "metrics": {
            "metric_status": "provisional",
            "matching": "raw_exact",
            "primary": {"name": "provisional_micro_f1", "score": score},
            "technical_failure_rate": 0,
        },
        "benchmark": {
            "measurement_status": "measured",
            "batch_policy": "serial",
            "environment": {"platform": "same", "machine": "same", "versions": {}},
            "warm_latency_p95_seconds": latency,
            "peak_ram_bytes": ram,
            "model_size_bytes": weights,
            "adapter_size_bytes": 0,
        },
        "synthetic_only": True,
    }


def test_pareto_tradeoff_does_not_pick_unmeasured_universal_winner():
    result = compare_runs(
        [measured("slow", 0.8, 3), measured("dominated", 0.5, 5), measured("fast", 0.7, 1)]
    )
    assert {x["id"] for x in result["rows"] if x["pareto"]} == {"slow", "fast"}
    assert result["winner"] is None and result["synthetic_only"] is True


def test_official_metric_direction_cannot_be_assumed():
    a, b = measured("a", -2, 1), measured("b", -1, 1)
    for run in [a, b]:
        run["metrics"]["primary"]["status"] = "official_plugin"
    with pytest.raises(ValueError, match="direction"):
        compare_runs([a, b])
    for run in [a, b]:
        run["metrics"]["primary"]["higher_is_better"] = False
    rows = compare_runs([a, b])["rows"]
    assert rows[0]["pareto"] is True and rows[1]["pareto"] is False


@pytest.mark.parametrize("field", ["data", "hardware", "failure", "metric"])
def test_noncomparable_or_failed_runs_are_not_ranked(field):
    a = measured("a", 0.5, 1)
    b = deepcopy(a)
    b["id"] = "b"
    if field == "data":
        b["canonical_data_sha256"] = "other-data"
    elif field == "hardware":
        b["benchmark"]["environment"]["machine"] = "other-machine"
    elif field == "failure":
        b["metrics"]["technical_failure_rate"] = 0.1
    else:
        b["metrics"]["primary"]["name"] = "different-metric"
    with pytest.raises(ValueError):
        compare_runs([a, b])
