#!/usr/bin/env python3
"""Run the qualified tiny SGLang fixture and retain a torch-profiler trace."""

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


def post(url, payload=None, timeout=300):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, method="POST", headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status, response.read().decode()


parser = argparse.ArgumentParser()
parser.add_argument("--fixture", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--graph", action="store_true")
args = parser.parse_args()
output = Path(args.output).resolve()
trace_dir = output / "traces"
output.mkdir(parents=True, exist_ok=True)
trace_dir.mkdir(parents=True, exist_ok=True)

# The server owns worker grandchildren. Reap them locally rather than leaving
# them with the container's non-reaping PID 1.
ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]

command = [
    sys.executable, "-m", "sglang.launch_server",
    "--model-path", args.fixture,
    "--tokenizer-path", args.fixture,
    "--served-model-name", "tiny-random-llama",
    "--host", "127.0.0.1", "--port", str(port),
    "--attention-backend", "triton", "--dtype", "float16",
    "--context-length", "128", "--max-total-tokens", "256",
    "--max-running-requests", "4", "--mem-fraction-static", "0.01",
    "--random-seed", "20260912",
]
if args.graph:
    command += ["--cuda-graph-max-bs-decode", "2", "--cuda-graph-bs-decode", "1", "2"]
else:
    command += ["--disable-cuda-graph"]

metadata = {
    "argv": sys.argv,
    "command": command,
    "mode": "graph" if args.graph else "eager",
}
env = os.environ.copy()
env["SGLANG_TORCH_PROFILER_DIR"] = str(trace_dir)
env["SGLANG_PROFILE_WITH_STACK"] = "false"
env["SGLANG_PROFILE_RECORD_SHAPES"] = "false"
started = time.time()
with (output / "server.log").open("wb") as log:
    proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    metadata["pid"] = proc.pid
    try:
        deadline = time.time() + 420
        while time.time() < deadline and proc.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        metadata["ready_after_seconds"] = round(time.time() - started, 3)
                        break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("server did not become ready")

        status, body = post(f"http://127.0.0.1:{port}/start_profile", {
            "num_steps": 10, "profile_by_stage": False,
            "activities": ["CPU", "GPU"], "with_stack": False,
            "record_shapes": False, "profile_id": args.output.split("/")[-1],
        }, timeout=60)
        metadata["start_profile"] = {"status": status, "body": body}
        request_payload = {
            "text": "hello world tiny llama rocm gpu test",
            "sampling_params": {"temperature": 0, "max_new_tokens": 48,
                                  "ignore_eos": True},
        }
        (output / "request.json").write_text(json.dumps(request_payload, indent=2) + "\n")
        status, body = post(f"http://127.0.0.1:{port}/generate", request_payload, timeout=300)
        metadata["generate_status"] = status
        (output / "response.json").write_text(body + "\n")
        # num_steps normally auto-stops. Explicit stop is only a fallback and
        # its response distinguishes an already-completed capture.
        time.sleep(3)
        try:
            status, body = post(f"http://127.0.0.1:{port}/stop_profile", timeout=60)
            metadata["stop_profile"] = {"status": status, "body": body}
        except urllib.error.HTTPError as exc:
            metadata["stop_profile"] = {"status": exc.code, "body": exc.read().decode()}
    except Exception as exc:
        metadata["error"] = repr(exc)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
        metadata["server_returncode"] = proc.returncode
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
        metadata["reaped_descendants"] = reaped
        metadata["elapsed_seconds"] = round(time.time() - started, 3)
        metadata["trace_files"] = [str(p) for p in trace_dir.glob("*")]

(output / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
raise SystemExit(0 if metadata.get("generate_status") == 200 and metadata["trace_files"] else 1)
