#!/usr/bin/env python3
"""Launch the qualified tiny SGLang fixture and kill a streaming client."""

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


if len(sys.argv) != 3:
    raise SystemExit("usage: run_disconnect_probe.py FIXTURE OUTPUT_DIR")

fixture = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
output.mkdir(parents=True, exist_ok=True)

# SGLang launches worker grandchildren. Become a child subreaper so shutdown
# does not leave them adopted by a non-reaping container PID 1.
PR_SET_CHILD_SUBREAPER = 36
ctypes.CDLL(None).prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]

python = sys.executable
command = [
    python,
    "-m",
    "sglang.launch_server",
    "--model-path",
    str(fixture),
    "--tokenizer-path",
    str(fixture),
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
    "4096",
    "--max-total-tokens",
    "4096",
    "--max-running-requests",
    "2",
    "--mem-fraction-static",
    "0.01",
    "--random-seed",
    "20260912",
    "--disable-cuda-graph",
]

request_body = {
    "model": "tiny-random-llama",
    "prompt": "one two three",
    "temperature": 0,
    "max_tokens": 3500,
    "stream": True,
}
request_path = output / "request.json"
request_path.write_text(json.dumps(request_body, indent=2) + "\n")

metadata = {
    "command": command,
    "port": port,
    "server_pid": None,
    "ready": False,
    "client_pid": None,
    "client_killed": False,
    "load_samples": [],
    "termination": None,
    "returncode": None,
}
server_log = output / "server.log"
client_output = output / "client-stream.txt"
started = time.monotonic()


def get_json(path, timeout=3):
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}{path}", timeout=timeout
    ) as response:
        return json.loads(response.read().decode())


def num_running_reqs(load):
    """Normalize legacy /get_load's per-DP-rank list response."""
    if isinstance(load, list):
        return sum(item.get("num_reqs", 0) for item in load)
    return load.get("num_running_reqs", load.get("num_reqs", 0))


with server_log.open("wb") as log:
    server_env = os.environ.copy()
    # The qualified fixture advertises 256 positions. This transport-only
    # probe needs enough decode time to kill the client before completion.
    server_env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    server = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=server_env,
    )
    metadata["server_pid"] = server.pid
    client = None
    try:
        deadline = time.monotonic() + 420
        while time.monotonic() < deadline and server.poll() is None:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2
                ) as response:
                    if response.status == 200:
                        metadata["ready"] = True
                        metadata["ready_after_seconds"] = round(
                            time.monotonic() - started, 3
                        )
                        break
            except Exception:
                time.sleep(1)

        if metadata["ready"]:
            curl_command = [
                "curl",
                "-sN",
                f"http://127.0.0.1:{port}/v1/completions",
                "-H",
                "Content-Type: application/json",
                "--data-binary",
                f"@{request_path}",
            ]
            with client_output.open("wb") as stream:
                client = subprocess.Popen(
                    curl_command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                metadata["client_pid"] = client.pid
                stream_deadline = time.monotonic() + 30
                while time.monotonic() < stream_deadline:
                    if client.poll() is not None or client_output.stat().st_size > 0:
                        break
                    time.sleep(0.05)
                metadata["bytes_before_kill"] = client_output.stat().st_size
                metadata["kill_after_seconds"] = round(time.monotonic() - started, 3)
                if client.poll() is None:
                    os.killpg(client.pid, signal.SIGKILL)
                    client.wait(timeout=5)
                    metadata["client_killed"] = True

            poll_started = time.monotonic()
            for _ in range(80):
                try:
                    load = get_json("/get_load")
                    metadata["load_samples"].append(
                        {
                            "seconds_after_kill": round(
                                time.monotonic() - poll_started, 3
                            ),
                            "response": load,
                            "num_running_reqs": num_running_reqs(load),
                        }
                    )
                    if num_running_reqs(load) == 0:
                        break
                except Exception as exc:
                    metadata["load_samples"].append(
                        {
                            "seconds_after_kill": round(
                                time.monotonic() - poll_started, 3
                            ),
                            "error": repr(exc),
                        }
                    )
                time.sleep(0.1)
            metadata["observed_after_kill_seconds"] = round(
                time.monotonic() - poll_started, 3
            )
            # Keep the server alive briefly to catch any orphan-output flood.
            time.sleep(3)
    finally:
        if client is not None and client.poll() is None:
            os.killpg(client.pid, signal.SIGKILL)
            client.wait(timeout=5)
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            metadata["termination"] = "SIGTERM to owned server process group"
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                metadata["termination"] += "; SIGKILL after 30 second timeout"
                server.wait(timeout=30)
        metadata["returncode"] = server.returncode
        reaped = []
        reap_deadline = time.monotonic() + 10
        while time.monotonic() < reap_deadline:
            try:
                child, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if child:
                reaped.append({"pid": child, "wait_status": status})
            else:
                time.sleep(0.1)
        metadata["reaped_descendants"] = reaped
        try:
            os.killpg(server.pid, 0)
            metadata["process_group_alive_after_cleanup"] = True
        except ProcessLookupError:
            metadata["process_group_alive_after_cleanup"] = False

log_text = server_log.read_text(errors="replace") if server_log.exists() else ""
metadata["deleted_state_error_count"] = log_text.count(
    "but the state was deleted in TokenizerManager"
)
metadata["abort_log_lines"] = [
    line
    for line in log_text.splitlines()
    if "abort" in line.lower() and "request" in line.lower()
][-30:]
metadata["elapsed_seconds"] = round(time.monotonic() - started, 3)
(output / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))

load_drained = any(
    sample.get("num_running_reqs") == 0
    for sample in metadata["load_samples"]
)
raise SystemExit(
    0
    if metadata["ready"]
    and metadata["client_killed"]
    and load_drained
    and metadata["deleted_state_error_count"] == 0
    else 1
)
