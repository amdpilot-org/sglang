#!/usr/bin/env python3
"""Launch a real SGLang server, wait for readiness, probe it, and reap it."""

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
parser.add_argument("--graph", action="store_true")
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)

# SGLang uses worker grandchildren. Become a Linux child subreaper so workers
# orphaned during shutdown can be wait(2)-reaped by this qualification runner.
PR_SET_CHILD_SUBREAPER = 36
ctypes.CDLL(None).prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]

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
]
if args.graph:
    command += ["--cuda-graph-max-bs-decode", "2", "--cuda-graph-bs-decode", "1", "2"]
else:
    command += ["--disable-cuda-graph"]

metadata = {
    "command": command,
    "port": port,
    "pid": None,
    "ready": False,
    "termination": None,
    "returncode": None,
}
log_path = output / "server.log"
started = time.time()
with log_path.open("wb") as log:
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    metadata["pid"] = process.pid
    try:
        deadline = time.time() + 420
        while time.time() < deadline:
            if process.poll() is not None:
                break
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2
                ) as response:
                    if response.status == 200:
                        metadata["ready"] = True
                        metadata["ready_after_seconds"] = round(
                            time.time() - started, 3
                        )
                        break
            except Exception:
                time.sleep(1)
        if metadata["ready"]:
            probe = Path(__file__).with_name("probe_model_validation.py")
            completed = subprocess.run(
                [sys.executable, str(probe), f"http://127.0.0.1:{port}", str(output)],
                text=True,
                capture_output=True,
                timeout=300,
            )
            metadata["probe_returncode"] = completed.returncode
            metadata["probe_stdout"] = completed.stdout
            metadata["probe_stderr"] = completed.stderr
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            metadata["termination"] = "SIGTERM to owned server process group"
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                metadata["termination"] += "; SIGKILL after 30 second timeout"
                process.wait(timeout=30)
        metadata["returncode"] = process.returncode
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
        metadata["elapsed_seconds"] = round(time.time() - started, 3)
        metadata["process_group_alive_after_cleanup"] = False
        try:
            os.killpg(process.pid, 0)
            metadata["process_group_alive_after_cleanup"] = True
        except ProcessLookupError:
            pass

(output / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
sys.exit(0 if metadata["ready"] and metadata.get("probe_returncode") == 0 else 1)
