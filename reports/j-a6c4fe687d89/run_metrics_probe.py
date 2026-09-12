#!/usr/bin/env python3
"""Launch the qualified tiny Llama fixture and capture live KV metrics."""

import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

fixture = Path(sys.argv[1])
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)

# SGLang launches worker grandchildren. Act as a child subreaper so the probe
# can clean up every process it owns without relying on container PID 1.
ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]

base_url = f"http://127.0.0.1:{port}"
command = [
    sys.executable,
    "-m",
    "sglang.launch_server",
    "--model-path",
    str(fixture),
    "--tokenizer-path",
    str(fixture),
    "--served-model-name",
    "tiny-random-llama",
    "--host",
    "127.0.0.1",
    "--port",
    str(port),
    "--attention-backend",
    "triton",
    "--dtype",
    "float16",
    "--context-length",
    "128",
    "--max-total-tokens",
    "256",
    "--max-running-requests",
    "4",
    "--mem-fraction-static",
    "0.01",
    "--random-seed",
    "20260912",
    "--disable-cuda-graph",
    "--enable-metrics",
]
metadata = {"command": command, "port": port, "ready": False}
request_body = {
    "text": "hello world",
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": 100,
        "ignore_eos": True,
    },
    "stream": True,
}
(output / "request.json").write_text(json.dumps(request_body, indent=2) + "\n")

request_result = {}
first_event = threading.Event()


def generate() -> None:
    request = urllib.request.Request(
        base_url + "/generate",
        data=json.dumps(request_body).encode(),
        headers={"Content-Type": "application/json"},
    )
    chunks = []
    with urllib.request.urlopen(request, timeout=180) as response:
        request_result["status"] = response.status
        while line := response.readline():
            chunks.append(line.decode())
            if line.startswith(b"data:"):
                first_event.set()
    request_result["body"] = "".join(chunks)


started = time.time()
with (output / "server.log").open("wb") as log:
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    metadata["pid"] = process.pid
    try:
        deadline = time.time() + 420
        while time.time() < deadline and process.poll() is None:
            try:
                with urllib.request.urlopen(
                    base_url + "/health", timeout=2
                ) as response:
                    if response.status == 200:
                        metadata["ready"] = True
                        break
            except Exception:
                time.sleep(1)
        if not metadata["ready"]:
            raise RuntimeError("server did not become ready")

        thread = threading.Thread(target=generate)
        thread.start()
        if not first_event.wait(timeout=120):
            raise RuntimeError("generation did not produce a streaming event")
        with urllib.request.urlopen(base_url + "/metrics", timeout=30) as response:
            metrics = response.read().decode()
        (output / "metrics.txt").write_text(metrics)
        thread.join(timeout=180)
        if thread.is_alive():
            raise RuntimeError("generation did not finish")
        (output / "response.json").write_text(
            json.dumps(request_result, indent=2) + "\n"
        )

        def value(name: str) -> float:
            prefix = name + "{"
            matches = [line for line in metrics.splitlines() if line.startswith(prefix)]
            if len(matches) != 1:
                raise AssertionError(f"expected one {name} sample, got {matches}")
            return float(matches[0].rsplit(" ", 1)[1])

        kv_usage = value("sglang:kv_cache_usage_perc")
        full_usage = value("sglang:full_token_usage")
        metadata["measurements"] = {
            "kv_cache_usage_perc": kv_usage,
            "full_token_usage": full_usage,
            "plain_model_reference_equal": kv_usage == full_usage,
            "bounded": 0.0 <= kv_usage <= 1.0,
        }
        if kv_usage != full_usage or not 0.0 <= kv_usage <= 1.0:
            raise AssertionError(metadata["measurements"])
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
        metadata["server_returncode"] = process.returncode
        metadata["elapsed_seconds"] = round(time.time() - started, 3)
        reaped = []
        reap_deadline = time.time() + 10
        while time.time() < reap_deadline:
            try:
                child, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if child:
                reaped.append({"pid": child, "wait_status": status})
            else:
                time.sleep(0.1)
        metadata["reaped_descendants"] = reaped
        (output / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

print(json.dumps(metadata, indent=2))
