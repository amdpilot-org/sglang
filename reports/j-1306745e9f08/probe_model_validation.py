#!/usr/bin/env python3
"""Issue-specific HTTP probes against an already-ready SGLang server."""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = sys.argv[1].rstrip("/")
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)


def post(name, path, body):
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        raw = response.read().decode()
        status = response.status
        headers = dict(response.headers)
    record = {
        "request": body,
        "status": status,
        "headers": headers,
        "raw_response": raw,
        "response": json.loads(raw),
    }
    (OUT / f"{name}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return record


results = {
    "chat_unknown": post(
        "chat_unknown",
        "/v1/chat/completions",
        {
            "model": "gpt-nonexistent-999",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0,
            "max_tokens": 1,
        },
    ),
    "completion_unknown": post(
        "completion_unknown",
        "/v1/completions",
        {
            "model": "gpt-nonexistent-999",
            "prompt": "Hello",
            "temperature": 0,
            "max_tokens": 1,
        },
    ),
    "chat_served": post(
        "chat_served",
        "/v1/chat/completions",
        {
            "model": "tiny-random-llama",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0,
            "max_tokens": 1,
        },
    ),
    "completion_served": post(
        "completion_served",
        "/v1/completions",
        {
            "model": "tiny-random-llama",
            "prompt": "Hello",
            "temperature": 0,
            "max_tokens": 1,
        },
    ),
}
summary = {
    name: {
        "status": result["status"],
        "response_model": result["response"].get("model"),
        "error": result["response"].get("error"),
    }
    for name, result in results.items()
}
(OUT / "probe-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
