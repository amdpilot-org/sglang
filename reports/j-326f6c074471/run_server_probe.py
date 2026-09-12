#!/usr/bin/env python3
"""Subreaper-enabled launcher for the private tiny-engine qualification."""

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

parser = argparse.ArgumentParser()
parser.add_argument("--fixture", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
command = [
    sys.executable, "-m", "sglang.launch_server", "--model-path", args.fixture,
    "--tokenizer-path", args.fixture, "--served-model-name", "tiny-random-llama",
    "--host", "127.0.0.1", "--port", str(port), "--attention-backend", "triton",
    "--dtype", "float16", "--context-length", "128", "--max-total-tokens", "256",
    "--max-running-requests", "4", "--mem-fraction-static", "0.01",
    "--random-seed", "20260912", "--disable-cuda-graph", "--schedule-policy", "fcfs",
]
metadata = {"command": command, "port": port, "ready": False}
with (output / "server.log").open("wb") as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    metadata["pid"] = process.pid
    try:
        deadline = time.time() + 420
        while time.time() < deadline and process.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    if response.status == 200:
                        metadata["ready"] = True
                        break
            except Exception:
                time.sleep(1)
        if metadata["ready"]:
            probe = Path(__file__).with_name("runtime_policy_probe.py")
            completed = subprocess.run(
                [sys.executable, str(probe), f"http://127.0.0.1:{port}", str(output)],
                text=True, capture_output=True, timeout=300,
            )
            metadata.update(probe_returncode=completed.returncode, probe_stdout=completed.stdout, probe_stderr=completed.stderr)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
        metadata["returncode"] = process.returncode
        metadata["reaped_descendants"] = []
        while True:
            try:
                child, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if not child:
                break
            metadata["reaped_descendants"].append({"pid": child, "status": status})
(output / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
sys.exit(0 if metadata["ready"] and metadata.get("probe_returncode") == 0 else 1)
