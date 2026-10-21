"""Fresh-process cold startup plus serial warm measurements on identical tickets."""

import json
import math
import subprocess
import sys
import time


def percentile(values, q):
    if not values or not 0 <= q <= 1:
        raise ValueError("percentile requires nonempty values and q in [0,1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def benchmark(input_path, config_path, repeats=10, warmup=1, timeout=120) -> dict:
    if repeats < 1 or warmup < 0 or timeout <= 0:
        raise ValueError("invalid benchmark budget")
    args = [
        sys.executable,
        "-m",
        "compactfacts._benchmark_worker",
        str(input_path),
        str(config_path),
        str(repeats),
        str(warmup),
    ]
    started = time.perf_counter()
    # communicate(timeout) also covers startup, unlike a blocking readline.
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as child:
        try:
            stdout, _stderr = child.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate()
            raise ValueError("benchmark worker timed out") from None
        if child.returncode:
            raise ValueError("benchmark worker failed; verify config/input/local model")
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        raise ValueError("invalid benchmark worker result") from None
    result["subprocess_round_trip_seconds"] = time.perf_counter() - started
    return result
