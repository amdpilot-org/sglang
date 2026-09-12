#!/usr/bin/env python3
"""Run the overlap/grammar abort sequence against a private tiny-Llama server."""

import argparse
import asyncio
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

import aiohttp


async def collect_stream(session, url, payload, events):
    async with session.post(url + "/generate", json=payload) as response:
        events.append({"kind": "response", "rid": payload["rid"], "status": response.status})
        async for raw in response.content:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            events.append(
                {
                    "kind": "chunk",
                    "rid": payload["rid"],
                    "after_cancel": bool(events_state["cancelled"]),
                    "body": json.loads(line[6:]),
                }
            )


events_state = {"cancelled": False}


async def probe(url, stagger_ms, cancel_delay_ms):
    events = []
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"color": {"type": "string", "enum": ["red", "blue"]}},
            "required": ["color"],
            "additionalProperties": False,
        }
    )
    primary = {
        "text": "Return a color as JSON",
        "rid": "grammar_primary",
        "stream": True,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "json_schema": schema,
        },
    }
    peer = {
        "text": "hello world",
        "rid": "overlap_peer",
        "stream": True,
        "sampling_params": {"temperature": 0, "max_new_tokens": 64},
    }
    async with aiohttp.ClientSession() as session:
        primary_task = asyncio.create_task(collect_stream(session, url, primary, events))
        await asyncio.sleep(stagger_ms / 1000)
        peer_task = asyncio.create_task(collect_stream(session, url, peer, events))
        await asyncio.sleep(cancel_delay_ms / 1000)
        async with session.post(
            url + "/abort_request", json={"rid": "grammar_primary"}
        ) as response:
            events_state["cancelled"] = True
            events.append({"kind": "cancel", "status": response.status})
        await asyncio.gather(primary_task, peer_task)
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stagger-ms", type=int, default=35)
    parser.add_argument("--cancel-delay-ms", type=int, default=80)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
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
        "4",
        "--mem-fraction-static",
        "0.01",
        "--disable-cuda-graph",
        "--random-seed",
        "20260912",
    ]
    metadata = {"command": command, "ready": False}
    with (output / "server.log").open("wb") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        metadata["pid"] = process.pid
        try:
            deadline = time.time() + 420
            while time.time() < deadline and process.poll() is None:
                try:
                    with urllib.request.urlopen(url + "/health", timeout=2) as response:
                        metadata["ready"] = response.status == 200
                        if metadata["ready"]:
                            break
                except Exception:
                    time.sleep(1)
            if metadata["ready"]:
                events = asyncio.run(probe(url, args.stagger_ms, args.cancel_delay_ms))
                (output / "events.json").write_text(json.dumps(events, indent=2) + "\n")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
            metadata["returncode"] = process.returncode
            reaped = []
            while True:
                try:
                    child, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if not child:
                    break
                reaped.append({"pid": child, "status": status})
            metadata["reaped"] = reaped
    (output / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0 if metadata["ready"] and (output / "events.json").exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
