#!/usr/bin/env python3
"""Issue-scoped streaming concurrency probe for an already-ready server."""

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request


def request_once(base: str, request_id: int) -> dict:
    body = {
        "model": "tiny-random-llama",
        "prompt": "the tiny llama test alpha beta gamma delta prompt " * 3
        + str(request_id),
        "temperature": 0,
        "max_tokens": 12,
        "stream": True,
    }
    request = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    event_times = []
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                event_times.append(time.perf_counter())
    ended = time.perf_counter()
    return {
        "status": response.status,
        "event_count": len(event_times),
        "e2e_ms": (ended - started) * 1000,
        "ttft_ms": (event_times[0] - started) * 1000,
        "stream_span_ms": (event_times[-1] - event_times[0]) * 1000,
    }


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[round((len(values) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[8, 64])
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    output = []
    for concurrency in args.concurrency:
        for round_index in range(args.rounds):
            with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                results = list(
                    pool.map(lambda i: request_once(args.base, i), range(concurrency))
                )
            assert all(item["status"] == 200 for item in results)
            assert all(item["event_count"] == results[0]["event_count"] for item in results)
            record = {
                "concurrency": concurrency,
                "round": round_index,
                "event_count": results[0]["event_count"],
                "e2e_p50_ms": statistics.median(x["e2e_ms"] for x in results),
                "e2e_p95_ms": percentile([x["e2e_ms"] for x in results], 0.95),
                "ttft_p50_ms": statistics.median(x["ttft_ms"] for x in results),
                "stream_span_p50_ms": statistics.median(
                    x["stream_span_ms"] for x in results
                ),
            }
            output.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    print("RESULT_JSON=" + json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
