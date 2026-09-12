#!/usr/bin/env python3
"""Exercise a real GPU server across an abrupt HTTP client disconnect."""

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

ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
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
    "--disable-cuda-graph",
]
environment = os.environ.copy()
environment["SGLANG_REQUEST_STATE_WAIT_TIMEOUT"] = "0.05"
metadata = {"command": command, "port": port, "ready": False}
log_path = output / "server.log"


def get(path, timeout=5):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as response:
        return response.status, response.read().decode()


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
                metadata["ready"] = get("/health")[0] == 200
                if metadata["ready"]:
                    break
            except Exception:
                time.sleep(1)
        if not metadata["ready"]:
            raise RuntimeError("server did not become ready")

        body = json.dumps(
            {
                "text": "hello world",
                "sampling_params": {"temperature": 0, "max_new_tokens": 100},
            }
        ).encode()
        request = (
            f"POST /generate HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode() + body
        with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
            client.sendall(request)
            client.shutdown(socket.SHUT_RDWR)
        metadata["disconnect"] = "request sent; TCP client closed without reading"

        time.sleep(5)
        metadata["process_alive_after_disconnect"] = process.poll() is None
        metadata["health_after_disconnect"] = get("/health")[0]
        followup = urllib.request.Request(
            f"http://127.0.0.1:{port}/generate",
            data=json.dumps(
                {
                    "text": "tiny llama",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(followup, timeout=120) as response:
            metadata["followup_status"] = response.status
            metadata["followup_response"] = json.loads(response.read())
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
        metadata["server_returncode"] = process.returncode
        while True:
            try:
                child, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if not child:
                time.sleep(0.1)

(output / "probe.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
