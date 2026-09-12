#!/usr/bin/env python3
"""Run repeated-prefix generation against one owned SGLang server."""

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

p = argparse.ArgumentParser()
p.add_argument("--fixture", required=True)
p.add_argument("--output", required=True)
p.add_argument("--mode", choices=("control", "l2", "file"), required=True)
a = p.parse_args()
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", a.fixture, "--tokenizer-path", a.fixture, "--served-model-name", "tiny-random-llama", "--host", "127.0.0.1", "--port", str(port), "--attention-backend", "triton", "--dtype", "float16", "--context-length", "256", "--max-total-tokens", "512", "--max-running-requests", "4", "--mem-fraction-static", "0.01", "--random-seed", "20260912", "--disable-cuda-graph", "--page-size", "1"]
if a.mode != "control":
    cmd += ["--enable-hierarchical-cache", "--hicache-ratio", "2"]
if a.mode == "file":
    cmd += ["--hicache-storage-backend", "file", "--hicache-storage-prefetch-policy", "wait_complete"]
meta = {"command": cmd, "mode": a.mode, "port": port, "ready": False}
log = (out / "server.log").open("wb")
proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
try:
    deadline = time.time() + 420
    while time.time() < deadline and proc.poll() is None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200: meta["ready"] = True; break
        except Exception: time.sleep(1)
    if meta["ready"]:
        prefix = " ".join((["hello", "world", "tiny", "llama", "rocm", "gpu", "test"] * 20)[:120])
        records = []
        for suffix in ("red", "red", "blue", "red"):
            body = {"text": prefix + " " + suffix, "sampling_params": {"temperature": 0, "max_new_tokens": 32, "ignore_eos": False}, "return_logprob": True}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as response: result = json.loads(response.read())
            records.append({"suffix": suffix, "text": result["text"], "meta_info": result["meta_info"]})
        (out / "requests.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
        same = [r["text"] for r in records if r["suffix"] == "red"]
        summary = {"red_outputs_identical": len(set(same)) == 1, "output_lengths": [r["meta_info"].get("completion_tokens") for r in records], "cached_tokens": [r["meta_info"].get("cached_tokens") for r in records], "cached_tokens_details": [r["meta_info"].get("cached_tokens_details") for r in records]}
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        meta["summary"] = summary
finally:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try: proc.wait(timeout=30)
        except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=30)
    log.close(); meta["returncode"] = proc.returncode
    while True:
        try:
            child, _ = os.waitpid(-1, os.WNOHANG)
            if not child: break
        except ChildProcessError: break
    (out / "run-metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
print(json.dumps(meta, indent=2))
sys.exit(0 if meta.get("ready") and meta.get("summary", {}).get("red_outputs_identical") else 1)
