#!/usr/bin/env python3
"""Launch the qualified tiny Llama and probe OpenAI auto-truncation."""

import argparse
import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--fixture", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)

# SGLang owns worker grandchildren. Act as their subreaper during shutdown.
ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)

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
    "32",
    "--max-total-tokens",
    "64",
    "--max-running-requests",
    "2",
    "--mem-fraction-static",
    "0.01",
    "--random-seed",
    "20260912",
    "--disable-cuda-graph",
    "--allow-auto-truncate",
]
metadata = {"command": command, "port": port, "ready": False}
request_body = {
    "model": "tiny-random-llama",
    "prompt": "one two three four five six seven eight nine",
    "temperature": 0,
    "max_tokens": 30,
}

with (output / "server.log").open("wb") as log:
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    metadata["pid"] = process.pid
    started = time.time()
    try:
        deadline = started + 420
        while time.time() < deadline and process.poll() is None:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2
                ) as response:
                    if response.status == 200:
                        metadata["ready"] = True
                        break
            except Exception:
                time.sleep(1)

        if metadata["ready"]:
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/completions",
                data=json.dumps(request_body).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    raw = response.read().decode()
                    status = response.status
            except urllib.error.HTTPError as error:
                raw = error.read().decode()
                status = error.code
            parsed = json.loads(raw)
            record = {
                "request": request_body,
                "status": status,
                "raw_response": raw,
                "response": parsed,
            }
            (output / "completion.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n"
            )
            usage = parsed.get("usage", {})
            metadata["assertions"] = {
                "http_200": status == 200,
                "requested_total_exceeds_context":
                    usage.get("prompt_tokens", 0) + request_body["max_tokens"] > 32,
                "actual_total_fits_context": usage.get("total_tokens", 33) <= 32,
                "completion_was_truncated":
                    usage.get("completion_tokens", 30) < request_body["max_tokens"],
            }
    finally:
        if process.poll() is None:
            # The launcher's SIGTERM crash-diagnostic path can signal its
            # parent. Kill only the owned server session after evidence is
            # flushed, then reap every descendant below this subreaper.
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
        metadata["returncode"] = process.returncode
        metadata["elapsed_seconds"] = round(time.time() - started, 3)
        reaped = []
        while True:
            try:
                child, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if not child:
                break
            reaped.append({"pid": child, "wait_status": status})
        metadata["reaped_descendants"] = reaped

(output / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
success = metadata["ready"] and all(metadata.get("assertions", {}).values())
sys.exit(0 if success else 1)
