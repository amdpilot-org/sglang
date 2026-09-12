#!/usr/bin/env python3
"""Launch the qualified tiny fixture and capture queue-metric HTTP evidence."""

import argparse
import ctypes
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path


def post(base, body):
    request = urllib.request.Request(
        base + "/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.status, response.read().decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fixture_manifest = Path(args.fixture) / "fixture-manifest.json"
    (output / "fixture-manifest.json").write_text(fixture_manifest.read_text())

    ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)  # Linux child subreaper.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
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
        "1",
        "--mem-fraction-static",
        "0.01",
        "--random-seed",
        "20260912",
        "--disable-cuda-graph",
        "--enable-metrics",
    ]
    metadata = {"command": command, "port": port, "ready": False}
    log_path = output / "server.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        metadata["pid"] = process.pid
        try:
            deadline = time.time() + 420
            while time.time() < deadline and process.poll() is None:
                try:
                    with urllib.request.urlopen(
                        base + "/health", timeout=2
                    ) as response:
                        if response.status == 200:
                            metadata["ready"] = True
                            break
                except Exception:
                    time.sleep(1)
            if not metadata["ready"]:
                raise RuntimeError("server did not become ready")

            requests = {}
            responses = {}

            def request(name, prompt):
                body = {
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 100,
                        "ignore_eos": True,
                    },
                }
                requests[name] = body
                responses[name] = post(base, body)

            first = threading.Thread(target=request, args=("first", "hello " * 20))
            second = threading.Thread(target=request, args=("second", "world " * 20))
            first.start()
            time.sleep(0.05)
            second.start()

            observations = []
            metric_re = re.compile(
                r'^sglang:avg_request_queue_latency\{[^}]*\}\s+([-+0-9.eE]+)$',
                re.MULTILINE,
            )
            queue_re = re.compile(
                r'^sglang:num_queue_reqs\{[^}]*\}\s+([-+0-9.eE]+)$', re.MULTILINE
            )
            poll_deadline = time.time() + 60
            while time.time() < poll_deadline and (
                first.is_alive() or second.is_alive()
            ):
                with urllib.request.urlopen(base + "/metrics", timeout=5) as response:
                    text = response.read().decode()
                metric = metric_re.search(text)
                queue = queue_re.search(text)
                observations.append(
                    {
                        "avg_request_queue_latency": (
                            float(metric.group(1)) if metric else None
                        ),
                        "num_queue_reqs": float(queue.group(1)) if queue else None,
                    }
                )
                if (
                    metric
                    and queue
                    and float(queue.group(1)) > 0
                    and float(metric.group(1)) > 0
                ):
                    (output / "metrics-while-queued.txt").write_text(text)
                    break
                time.sleep(0.02)
            first.join(180)
            second.join(180)
            (output / "requests.json").write_text(
                json.dumps(requests, indent=2) + "\n"
            )
            (output / "responses.json").write_text(
                json.dumps(
                    {
                        key: {"status": value[0], "raw_response": value[1]}
                        for key, value in responses.items()
                    },
                    indent=2,
                )
                + "\n"
            )
            metadata["responses"] = {
                key: value[0] for key, value in responses.items()
            }
            metadata["observations"] = observations
            metadata["observed_positive_queued_latency"] = any(
                item["num_queue_reqs"] is not None
                and item["num_queue_reqs"] > 0
                and item["avg_request_queue_latency"] is not None
                and item["avg_request_queue_latency"] > 0
                for item in observations
            )
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(30)
            metadata["returncode"] = process.returncode
            metadata["reaped_descendants"] = []
            while True:
                try:
                    child, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if not child:
                    break
                metadata["reaped_descendants"].append([child, status])

    (output / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0 if metadata.get("observed_positive_queued_latency") else 1


if __name__ == "__main__":
    raise SystemExit(main())
