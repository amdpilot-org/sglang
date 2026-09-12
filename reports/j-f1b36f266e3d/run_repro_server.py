#!/usr/bin/env python3
"""Launch and reap one source-checkout server for the issue reproduction."""

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
parser.add_argument("--delay", type=float, default=0)
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
command = [
    sys.executable, "-m", "sglang.launch_server",
    "--model-path", args.fixture, "--tokenizer-path", args.fixture,
    "--host", "127.0.0.1", "--port", str(port),
    "--attention-backend", "triton", "--dtype", "float16",
    "--context-length", "256", "--max-total-tokens", "512",
    "--max-running-requests", "4", "--mem-fraction-static", "0.01",
    "--disable-cuda-graph", "--enable-streaming-session",
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
            client = Path(__file__).with_name("repro_streaming_disconnect.py")
            completed = subprocess.run(
                [sys.executable, str(client), f"http://127.0.0.1:{port}",
                 str(output / "result.json"), "--post-disconnect-delay", str(args.delay)],
                text=True, capture_output=True, timeout=180,
            )
            metadata.update(client_exit_code=completed.returncode,
                            client_stdout=completed.stdout, client_stderr=completed.stderr)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
        metadata["server_exit_code"] = process.returncode
        reaped = []
        while True:
            try:
                child, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if not child:
                break
            reaped.append({"pid": child, "status": status})
        metadata["reaped_descendants"] = reaped
(output / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
raise SystemExit(0 if metadata.get("client_exit_code") == 0 else 1)
