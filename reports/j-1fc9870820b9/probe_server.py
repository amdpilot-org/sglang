#!/usr/bin/env python3
"""Run protocol probes against an already-ready SGLang HTTP server."""

import json
import sys
import urllib.request
from pathlib import Path


BASE = sys.argv[1].rstrip("/")
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)


def post(name, path, body, stream=False):
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read().decode()
        status = response.status
        headers = dict(response.headers)
    record = {"request": body, "status": status, "headers": headers, "raw_response": raw}
    if not stream:
        parsed = json.loads(raw)
        record["response"] = parsed
        record["response_schema"] = describe(parsed)
    else:
        events = []
        for line in raw.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))
        record["event_count"] = len(events)
        record["events"] = events
        record["response_schema"] = describe(events)
    (OUT / f"{name}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def describe(value):
    if isinstance(value, dict):
        return {key: describe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [describe(value[0])] if value else []
    return type(value).__name__


results = {}
results["generate"] = post("generate", "/generate", {
    "text": "hello world", "sampling_params": {"temperature": 0, "max_new_tokens": 8}
})
results["completion"] = post("completion", "/v1/completions", {
    "model": "tiny-random-llama", "prompt": "tiny llama", "temperature": 0,
    "max_tokens": 8,
})
results["batch"] = post("batch", "/generate", {
    "text": ["red blue", "one two three"],
    "sampling_params": {"temperature": 0, "max_new_tokens": 6},
})
results["stream"] = post("stream", "/v1/completions", {
    "model": "tiny-random-llama", "prompt": "stream test", "temperature": 0,
    "max_tokens": 8, "stream": True,
}, stream=True)
(OUT / "probe-summary.json").write_text(json.dumps({
    "http_statuses": {key: item["status"] for key, item in results.items()},
    "stream_event_count": results["stream"]["event_count"],
}, indent=2, sort_keys=True) + "\n")
