#!/usr/bin/env python3
"""Run the qualified tiny Llama fixture through CPU torch-native LoRA."""

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


fixture = Path(sys.argv[1])
adapter = Path(sys.argv[2])
output = Path(sys.argv[3])
output.mkdir(parents=True, exist_ok=True)

# SGLang launches worker grandchildren. Act as a subreaper so cleanup can wait
# for descendants instead of leaving them to the container's PID 1.
ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]

command = [
    sys.executable,
    "-m",
    "sglang.launch_server",
    "--model-path",
    str(fixture),
    "--tokenizer-path",
    str(fixture),
    "--host",
    "127.0.0.1",
    "--port",
    str(port),
    "--device",
    "cpu",
    "--attention-backend",
    "torch_native",
    "--dtype",
    "bfloat16",
    "--context-length",
    "128",
    "--max-total-tokens",
    "256",
    "--max-running-requests",
    "2",
    "--disable-cuda-graph",
    "--enable-lora",
    "--lora-backend",
    "torch_native",
    "--lora-paths",
    f"tiny={adapter}",
]
environment = os.environ.copy()
environment["SGLANG_USE_CPU_ENGINE"] = "1"
metadata = {"command": command, "port": port, "ready": False}
log_path = output / "server.log"

with log_path.open("wb") as log:
    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )
    metadata["pid"] = process.pid
    try:
        deadline = time.time() + 420
        while time.time() < deadline and process.poll() is None:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2
                ) as response:
                    metadata["ready"] = response.status == 200
                if metadata["ready"]:
                    break
            except Exception:
                time.sleep(1)

        if metadata["ready"]:
            request_body = {
                "text": "hello world",
                "lora_path": "tiny",
                "sampling_params": {"temperature": 0, "max_new_tokens": 4},
            }
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/generate",
                data=json.dumps(request_body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                raw_response = response.read().decode()
                metadata["http_status"] = response.status
            (output / "request.json").write_text(json.dumps(request_body, indent=2))
            (output / "response.json").write_text(raw_response)
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
            if child == 0:
                break
            metadata["reaped_descendants"].append(
                {"pid": child, "wait_status": status}
            )

(output / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
sys.exit(0 if metadata.get("http_status") == 200 else 1)
