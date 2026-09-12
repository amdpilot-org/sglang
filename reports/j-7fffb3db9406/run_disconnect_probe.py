#!/usr/bin/env python3
"""Run a real tiny-model streaming disconnect probe and retain raw evidence."""

import argparse
import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def get_json(url):
    with urllib.request.urlopen(url, timeout=2) as response:
        return response.status, json.loads(response.read().decode())


parser = argparse.ArgumentParser()
parser.add_argument("--fixture", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)

# SGLang workers are grandchildren. Reap them after terminating our process group.
ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)
with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]

command = [
    sys.executable,
    "-m",
    "sglang.launch_server",
    "--model-path",
    args.fixture,
    "--tokenizer-path",
    args.fixture,
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
]
record = {"command": command, "port": port, "ready": False}
log_path = output / "server.log"
started = time.time()

with log_path.open("wb") as log:
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    record["pid"] = process.pid
    try:
        deadline = time.time() + 420
        while time.time() < deadline and process.poll() is None:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2
                ) as response:
                    if response.status == 200:
                        record["ready"] = True
                        break
            except Exception:
                time.sleep(1)
        if not record["ready"]:
            raise RuntimeError("server did not become ready")

        body = json.dumps(
            {
                "text": "hello world",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 100,
                    "ignore_eos": True,
                },
                "stream": True,
            }
        ).encode()
        request_head = (
            f"POST /generate HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
        ).encode()
        with socket.create_connection(("127.0.0.1", port), timeout=10) as client:
            client.sendall(request_head + body)
            chunks = []
            response_deadline = time.time() + 30
            while time.time() < response_deadline:
                chunks.append(client.recv(4096))
                if b"data:" in b"".join(chunks):
                    break
            first_bytes = b"".join(chunks)
        record["disconnect_after_bytes"] = len(first_bytes)
        record["saw_sse_data_event"] = b"data:" in first_bytes

        samples = []
        drain_deadline = time.time() + 12
        while time.time() < drain_deadline:
            try:
                status, load = get_json(f"http://127.0.0.1:{port}/get_load")
                samples.append({"t": round(time.time() - started, 3), "status": status, "body": load})
                workers = load if isinstance(load, list) else [load]
                if all(
                    worker.get("num_running_reqs", worker.get("num_reqs")) == 0
                    for worker in workers
                ):
                    break
            except Exception as exc:
                samples.append({"t": round(time.time() - started, 3), "error": repr(exc)})
            time.sleep(0.25)
        record["load_samples"] = samples
        last_load = samples[-1].get("body") if samples else None
        last_workers = last_load if isinstance(last_load, list) else [last_load]
        record["drained"] = bool(last_load) and all(
            worker.get("num_running_reqs", worker.get("num_reqs")) == 0
            for worker in last_workers
        )
        # Allow any stale output/error line to be flushed before inspecting the log.
        time.sleep(2)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
        record["returncode"] = process.returncode
        while True:
            try:
                child, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if not child:
                break

log_text = log_path.read_text(errors="replace")
record["deleted_state_error_count"] = log_text.count(
    "but the state was deleted in TokenizerManager"
)
record["gpu_execution_evidence"] = [
    line
    for line in log_text.splitlines()
    if "Prefill batch" in line or "Decode batch" in line or "HIP" in line
][-20:]
record["elapsed_seconds"] = round(time.time() - started, 3)
(output / "disconnect-result.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
sys.exit(0 if record["ready"] and record["drained"] and record["deleted_state_error_count"] == 0 else 1)
