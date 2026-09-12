#!/usr/bin/env python3
"""Constrained one-GPU NIXL P/D role-replacement probe with owned cleanup."""

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

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evidence" / "single_gpu_nixl_skip_warmup"
OUT.mkdir(parents=True, exist_ok=True)
FIXTURE = Path(os.environ.get(
    "SGLANG_ISSUE_FIXTURE",
    "/tmp/amdpilot-repo-j-3ec1a4e9c131/tiny-random-llama",
))

# SGLang workers are grandchildren. Adopt them so cleanup does not leak zombies.
ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)


def ports(count):
    held = []
    values = []
    for _ in range(count):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        held.append(sock)
        values.append(sock.getsockname()[1])
    for sock in held:
        sock.close()
    return values


prefill_port, decode_port, router_port, bootstrap_port, prefill_nccl, decode_nccl = ports(6)
children = {}
logs = {}
events = []


def start(name, command, extra_env=None):
    log = (OUT / f"{name}.log").open("ab", buffering=0)
    env = os.environ.copy()
    env.update({
        "SGLANG_DISAGGREGATION_NIXL_BACKEND": "UCX",
        "SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS": json.dumps({"num_threads": "2"}),
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "20",
        "PYTHONUNBUFFERED": "1",
    })
    if extra_env:
        env.update(extra_env)
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, env=env,
        start_new_session=True,
    )
    children[name] = process
    logs[name] = log
    events.append({"event": "start", "name": name, "pid": process.pid, "command": command})
    return process


def stop(name):
    process = children.pop(name, None)
    if process is None:
        return
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=25)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=20)
    events.append({"event": "stop", "name": name, "pid": process.pid, "returncode": process.returncode})
    logs.pop(name).close()


def get(path, timeout=2):
    with urllib.request.urlopen(path, timeout=timeout) as response:
        return response.status


def wait_ready(name, url, timeout=360):
    deadline = time.time() + timeout
    while time.time() < deadline:
        process = children[name]
        if process.poll() is not None:
            raise RuntimeError(f"{name} exited with {process.returncode}")
        try:
            if get(url) == 200:
                events.append({"event": "ready", "name": name, "url": url})
                return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"{name} not ready: {url}")


common = [
    sys.executable, "-m", "sglang.launch_server",
    "--model-path", str(FIXTURE), "--tokenizer-path", str(FIXTURE),
    "--served-model-name", "tiny-random-llama", "--host", "127.0.0.1",
    "--attention-backend", "triton", "--dtype", "float16",
    "--context-length", "128", "--max-total-tokens", "256",
    "--max-running-requests", "4", "--mem-fraction-static", "0.01",
    "--random-seed", "20260912", "--disable-cuda-graph",
    "--skip-server-warmup", "--num-reserved-decode-tokens", "32",
    "--disaggregation-transfer-backend", "nixl", "--base-gpu-id", "0",
]


def start_prefill(suffix=""):
    start("prefill" + suffix, common + [
        "--port", str(prefill_port), "--disaggregation-mode", "prefill",
        "--disaggregation-bootstrap-port", str(bootstrap_port),
        "--nccl-port", str(prefill_nccl),
    ])
    wait_ready("prefill" + suffix, f"http://127.0.0.1:{prefill_port}/health")


def start_decode(suffix=""):
    start("decode" + suffix, common + [
        "--port", str(decode_port), "--disaggregation-mode", "decode",
        "--disaggregation-bootstrap-port", str(bootstrap_port),
        "--nccl-port", str(decode_nccl),
    ])
    wait_ready("decode" + suffix, f"http://127.0.0.1:{decode_port}/health")


def start_router():
    start("router", [
        sys.executable, "-m", "sglang_router.launch_router",
        "--pd-disaggregation", "--mini-lb", "--host", "127.0.0.1",
        "--port", str(router_port),
        "--prefill", f"http://127.0.0.1:{prefill_port}", str(bootstrap_port),
        "--decode", f"http://127.0.0.1:{decode_port}",
    ])
    wait_ready("router", f"http://127.0.0.1:{router_port}/health", timeout=120)


def request(label):
    body = json.dumps({
        "text": "hello world",
        "sampling_params": {"temperature": 0, "max_new_tokens": 8},
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{router_port}/generate", data=body,
        headers={"Content-Type": "application/json"},
    )
    record = {"label": label, "request": json.loads(body), "started": time.time()}
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            raw = response.read().decode()
            record.update({"status": response.status, "response": json.loads(raw)})
    except Exception as exc:
        record.update({"error": repr(exc)})
        if isinstance(exc, urllib.error.HTTPError):
            record["status"] = exc.code
            record["body"] = exc.read().decode(errors="replace")
    record["elapsed_seconds"] = round(time.time() - record.pop("started"), 3)
    (OUT / f"{label}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    events.append({"event": "request", **record})
    return record


result = {"ports": {
    "prefill": prefill_port, "decode": decode_port, "router": router_port,
    "bootstrap": bootstrap_port,
}, "fixture": str(FIXTURE), "phases": {}}
try:
    start_prefill()
    start_decode()
    start_router()
    result["phases"]["baseline"] = request("baseline")

    stop("decode")
    start_decode("_replacement")
    result["phases"]["after_decode_replacement"] = request("after_decode_replacement")

    stop("decode_replacement")
    stop("prefill")
    start_prefill("_pair_restart")
    start_decode("_pair_restart")
    result["phases"]["after_pair_restart"] = request("after_pair_restart")

    stop("prefill_pair_restart")
    start_prefill("_replacement")
    result["phases"]["after_prefill_replacement"] = request("after_prefill_replacement")
finally:
    for name in list(children):
        stop(name)
    reaped = []
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid:
            reaped.append({"pid": pid, "status": status})
        else:
            time.sleep(0.1)
    result["events"] = events
    result["reaped_descendants"] = reaped
    (OUT / "probe-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
