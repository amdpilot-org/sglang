#!/usr/bin/env python3
"""Run the issue's serving/engine metric comparison on the tiny GPU fixture."""

import argparse
import ctypes
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--fixture", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)

# SGLang launches worker grandchildren. Act as their subreaper for clean shutdown.
ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]

server_command = [
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
    "512",
    "--max-running-requests",
    "4",
    "--mem-fraction-static",
    "0.01",
    "--random-seed",
    "20260912",
    "--disable-cuda-graph",
    "--decode-log-interval",
    "10",
]
bench_command = [
    sys.executable,
    "-m",
    "sglang.benchmark.serving",
    "--backend",
    "sglang",
    "--dataset-name",
    "random",
    "--random-range-ratio",
    "1.0",
    "--random-input-len",
    "32",
    "--random-output-len",
    "64",
    "--num-prompts",
    "4",
    "--max-concurrency",
    "4",
    "--host",
    "127.0.0.1",
    "--port",
    str(port),
    "--model",
    args.fixture,
    "--tokenizer",
    args.fixture,
]

metadata = {"server_command": server_command, "bench_command": bench_command}
server_log = output / "server.log"
with server_log.open("wb") as log:
    process = subprocess.Popen(
        server_command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    metadata["server_pid"] = process.pid
    try:
        deadline = time.time() + 420
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"server exited early with {process.returncode}")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2
                ) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("server readiness timed out")

        bench = subprocess.run(
            bench_command, text=True, capture_output=True, timeout=300
        )
        (output / "bench.stdout").write_text(bench.stdout)
        (output / "bench.stderr").write_text(bench.stderr)
        metadata["bench_returncode"] = bench.returncode
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
        reaped = []
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                child, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if child:
                reaped.append({"pid": child, "status": status})
            else:
                time.sleep(0.1)
        metadata["server_returncode"] = process.returncode
        metadata["reaped_descendants"] = reaped

log_text = server_log.read_text(errors="replace")
bench_text = (output / "bench.stdout").read_text(errors="replace")
engine_rates = [
    float(value)
    for value in re.findall(r"gen throughput \(token/s\): ([0-9.]+)", log_text)
]

def metric(label):
    match = re.search(rf"^{re.escape(label)}\s+([0-9.]+)", bench_text, re.MULTILINE)
    return float(match.group(1)) if match else None


summary = {
    "metadata": metadata,
    "bench": {
        "mean_tpot_ms": metric("Mean TPOT (ms):"),
        "median_tpot_ms": metric("Median TPOT (ms):"),
        "output_throughput_tok_s": metric("Output token throughput (tok/s):"),
        "concurrency": metric("Concurrency:"),
    },
    "engine_decode_window_rates_tok_s": engine_rates,
}
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
sys.exit(metadata.get("bench_returncode", 1))
