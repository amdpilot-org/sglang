#!/usr/bin/env python3
"""Launch one backend server, benchmark it, and reap all owned descendants."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=("python", "rust"), required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        sys.executable, "-m", "sglang.launch_server", "--model-path", args.fixture,
        "--tokenizer-path", args.fixture, "--served-model-name", "tiny-random-llama",
        "--host", "127.0.0.1", "--port", str(port), "--attention-backend", "triton",
        "--dtype", "float16", "--context-length", "128", "--max-total-tokens", "4096",
        "--max-running-requests", "64", "--mem-fraction-static", "0.02",
        "--random-seed", "20260912", "--disable-cuda-graph",
    ]
    env = os.environ.copy()
    env["SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND"] = args.backend
    metadata = {"backend": args.backend, "command": command, "ready": False}
    started = time.time()
    with (output / "server.log").open("wb") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env,
            start_new_session=True,
        )
        try:
            for _ in range(420):
                if process.poll() is not None:
                    break
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
                probe = Path(__file__).with_name("bench_streaming_server.py")
                run = subprocess.run(
                    [sys.executable, str(probe), f"http://127.0.0.1:{port}"],
                    text=True, capture_output=True, timeout=600,
                )
                (output / "benchmark.log").write_text(run.stdout + run.stderr)
                metadata["benchmark_returncode"] = run.returncode
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
            while True:
                try:
                    child, _ = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if not child:
                    time.sleep(0.1)
            metadata["server_returncode"] = process.returncode
            metadata["elapsed_seconds"] = round(time.time() - started, 3)
    (output / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0 if metadata["ready"] and metadata.get("benchmark_returncode") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
