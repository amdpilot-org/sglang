#!/usr/bin/env python3
"""Launch the tiny fixture and verify environment-backed HTTP auth safely."""

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


def request(url, *, key=None, method="GET", body=None):
    headers = {}
    if key is not None:
        headers["Authorization"] = f"Bearer {key}"
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()
    except urllib.error.URLError:
        return 0, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    api_key = "canary-http-user-2f8a"
    admin_api_key = "canary-http-admin-6c31"
    env = os.environ.copy()
    env["SGLANG_API_KEY"] = api_key
    env["SGLANG_ADMIN_API_KEY"] = admin_api_key

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
        "128",
        "--max-total-tokens",
        "256",
        "--max-running-requests",
        "4",
        "--mem-fraction-static",
        "0.01",
        "--disable-cuda-graph",
    ]
    log_path = output / "server.log"
    evidence = {"command": command, "ready": False}
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            cmdline = Path(f"/proc/{process.pid}/cmdline").read_bytes()
            evidence["argv_has_api_key"] = api_key.encode() in cmdline
            evidence["argv_has_admin_api_key"] = admin_api_key.encode() in cmdline
            deadline = time.time() + 420
            while time.time() < deadline and process.poll() is None:
                status, _ = request(f"http://127.0.0.1:{port}/health")
                if status == 200:
                    evidence["ready"] = True
                    break
                time.sleep(1)
            if evidence["ready"]:
                base = f"http://127.0.0.1:{port}"
                evidence["models_without_key_status"] = request(base + "/v1/models")[0]
                evidence["models_with_api_key_status"] = request(
                    base + "/v1/models", key=api_key
                )[0]
                info_status, info_body = request(base + "/server_info", key=api_key)
                evidence["server_info_status"] = info_status
                evidence["server_info_has_api_key"] = api_key in info_body
                evidence["server_info_has_admin_api_key"] = admin_api_key in info_body
                info = json.loads(info_body)
                evidence["server_info_api_key"] = info.get("api_key")
                evidence["server_info_admin_api_key"] = info.get("admin_api_key")
                evidence["admin_endpoint_with_api_key_status"] = request(
                    base + "/set_internal_state", key=api_key, method="POST", body={}
                )[0]
                evidence["admin_endpoint_with_admin_key_status"] = request(
                    base + "/set_internal_state",
                    key=admin_api_key,
                    method="POST",
                    body={},
                )[0]
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
                    break
            evidence["returncode"] = process.returncode

    log_bytes = log_path.read_bytes()
    evidence["log_has_api_key"] = api_key.encode() in log_bytes
    evidence["log_has_admin_api_key"] = admin_api_key.encode() in log_bytes
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    expected = {
        "ready": True,
        "argv_has_api_key": False,
        "argv_has_admin_api_key": False,
        "models_without_key_status": 401,
        "models_with_api_key_status": 200,
        "server_info_status": 200,
        "server_info_has_api_key": False,
        "server_info_has_admin_api_key": False,
        "server_info_api_key": "<redacted>",
        "server_info_admin_api_key": "<redacted>",
        "admin_endpoint_with_api_key_status": 401,
        "admin_endpoint_with_admin_key_status": 400,
        "log_has_api_key": False,
        "log_has_admin_api_key": False,
    }
    return (
        0 if all(evidence.get(key) == value for key, value in expected.items()) else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
