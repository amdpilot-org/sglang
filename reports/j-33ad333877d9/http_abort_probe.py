#!/usr/bin/env python3
"""Exercise a live streaming request and the public abort endpoint."""

import json
import sys
import threading
import time
from pathlib import Path

import requests


base_url = sys.argv[1]
output = Path(sys.argv[2])
rid = "j33_abort_live"
stream_result = {"events": [], "error": None}
first_event = threading.Event()


def consume_stream():
    try:
        response = requests.post(
            f"{base_url}/generate",
            json={
                "text": "abort primary request with long decode trajectory",
                "sampling_params": {
                    "max_new_tokens": 120,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": 1,
                    "min_p": 0.0,
                    "ignore_eos": True,
                    "n": 1,
                    "sampling_seed": 0,
                },
                "stream": True,
                "rid": rid,
            },
            stream=True,
            timeout=60,
        )
        stream_result["status_code"] = response.status_code
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            stream_result["events"].append(line)
            first_event.set()
    except Exception as exc:
        stream_result["error"] = repr(exc)
        first_event.set()


thread = threading.Thread(target=consume_stream)
thread.start()
if not first_event.wait(timeout=30):
    raise RuntimeError("stream did not produce its first event")

abort_response = requests.post(
    f"{base_url}/abort_request",
    json={"rid": rid, "abort_all": False},
    timeout=10,
)
thread.join(timeout=30)

result = {
    "rid": rid,
    "abort_status_code": abort_response.status_code,
    "abort_body": abort_response.text,
    "stream_thread_alive": thread.is_alive(),
    "stream": stream_result,
}
(output / "http-result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))

if abort_response.status_code != 200 or thread.is_alive() or stream_result["error"]:
    raise SystemExit(1)
if not any('"type":"abort"' in event.replace(" ", "") for event in stream_result["events"]):
    raise SystemExit("stream did not report an abort finish reason")
