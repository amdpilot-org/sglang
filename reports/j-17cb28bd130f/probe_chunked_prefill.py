#!/usr/bin/env python3
"""Exercise four staggered chunked-prefill requests and sample scheduler load."""

import concurrent.futures
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

base_url, output_arg = sys.argv[1:]
output = Path(output_arg)
stop = threading.Event()
loads = []


def get_json(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        base_url + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def sample_loads():
    while not stop.is_set():
        try:
            loads.append({"time": time.time(), "load": get_json("/v1/loads")})
        except Exception as error:
            loads.append({"time": time.time(), "error": repr(error)})
        stop.wait(0.05)


words = ["alpha", "beta", "gamma", "delta", "red", "blue", "green"]
prompt = " ".join(words * 10)


def call(index, delay):
    time.sleep(delay)
    started = time.time()
    response = get_json(
        "/v1/completions",
        {
            "model": "tiny-random-llama",
            "prompt": f"request tok{index:03d} {prompt}",
            "max_tokens": 30,
            "temperature": 0,
        },
    )
    return {
        "index": index,
        "delay": delay,
        "started": started,
        "finished": time.time(),
        "usage": response.get("usage"),
    }


thread = threading.Thread(target=sample_loads, daemon=True)
thread.start()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(call, range(4), (0, 0.05, 0.1, 0.2)))
stop.set()
thread.join()

document = {"requests": results, "load_samples": loads}
output.write_text(json.dumps(document, indent=2) + "\n")
print(json.dumps(document, indent=2))
