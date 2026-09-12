#!/usr/bin/env python3
"""Exercise queued streaming-session abort recovery against a real server."""

import argparse
import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def post(base, path, body, timeout=120):
    request = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.fixture,
        "--tokenizer-path",
        args.fixture,
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
        "1",
        "--mem-fraction-static",
        "0.01",
        "--disable-cuda-graph",
        "--enable-streaming-session",
    ]
    metadata = {"command": command, "ready": False}
    log_path = output / "server.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        metadata["pid"] = process.pid
        try:
            deadline = time.time() + 420
            while time.time() < deadline and process.poll() is None:
                try:
                    with urllib.request.urlopen(base + "/health", timeout=2) as response:
                        if response.status == 200:
                            metadata["ready"] = True
                            break
                except Exception:
                    time.sleep(1)
            if not metadata["ready"]:
                raise RuntimeError("server did not become ready")

            records = {}
            records["open"] = post(
                base, "/open_session", {"capacity_of_str_len": 1000, "streaming": True}
            )
            session_id = records["open"][1]
            records["first"] = post(
                base,
                "/generate",
                {
                    "input_ids": [4, 5],
                    "session_params": {"id": session_id, "rid": None},
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                },
            )
            first_rid = records["first"][1]["meta_info"]["id"]

            def request(name, body):
                records[name] = post(base, "/generate", body)

            blocker = threading.Thread(
                target=request,
                args=(
                    "blocker",
                    {
                        "input_ids": [6, 7],
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 60,
                            "ignore_eos": True,
                        },
                    },
                ),
            )
            queued_rid = "queued-streaming-turn"
            queued = threading.Thread(
                target=request,
                args=(
                    "queued",
                    {
                        "input_ids": [8, 9],
                        "rid": queued_rid,
                        "session_params": {"id": session_id, "rid": first_rid},
                        "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                    },
                ),
            )
            blocker.start()
            time.sleep(0.25)
            queued.start()
            time.sleep(0.25)
            records["abort"] = post(base, "/abort_request", {"rid": queued_rid})
            queued.join(120)
            blocker.join(120)
            records["next"] = post(
                base,
                "/generate",
                {
                    "input_ids": [10, 11],
                    "session_params": {"id": session_id, "rid": first_rid},
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                },
            )
            (output / "requests.json").write_text(
                json.dumps(records, indent=2, sort_keys=True) + "\n"
            )
            queued_reason = records["queued"][1]["meta_info"]["finish_reason"]
            passed = (
                records["queued"][0] == 200
                and queued_reason["type"] == "abort"
                and records["queued"][1]["output_ids"] == []
                and records["next"][0] == 200
                and "error" not in records["next"][1]
            )
            metadata["passed"] = passed
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(30)
            metadata["returncode"] = process.returncode
            reap_deadline = time.time() + 10
            while time.time() < reap_deadline:
                try:
                    child, _ = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if not child:
                    time.sleep(0.1)
    (output / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0 if metadata.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
