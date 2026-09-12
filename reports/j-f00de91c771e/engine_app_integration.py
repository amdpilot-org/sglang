#!/usr/bin/env python3
"""Exercise a real Engine through the in-process ASGI factory."""

import ctypes
import json
import os
import time
from pathlib import Path

import torch
from fastapi.testclient import TestClient

import sglang
from sglang.srt.entrypoints.http_server import build_app, init_app_state

ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER


def main():
    fixture = Path(os.environ["SGLANG_QUAL_FIXTURE"])
    output = Path(os.environ["SGLANG_INTEGRATION_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "fixture": str(fixture),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0),
    }
    engine = None
    started = time.time()
    try:
        engine = sglang.Engine(
            model_path=str(fixture),
            tokenizer_path=str(fixture),
            served_model_name="tiny-random-llama",
            attention_backend="triton",
            dtype="float16",
            context_length=128,
            max_total_tokens=256,
            max_running_requests=4,
            mem_fraction_static=0.01,
            random_seed=20260912,
            disable_cuda_graph=True,
        )
        app = build_app(engine.server_args)
        init_app_state(engine, app.state, engine.server_args)
        with TestClient(app) as client:
            model_info = client.get("/model_info")
            generation = client.post(
                "/generate",
                json={
                    "text": "hello world",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                },
            )
        record.update(
            model_info_status=model_info.status_code,
            model_info=model_info.json(),
            generation_status=generation.status_code,
            generation=generation.json(),
            output_token_count=generation.json()["meta_info"]["completion_tokens"],
        )
    finally:
        if engine is not None:
            engine.shutdown()
        reaped = []
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if not pid:
                break
            reaped.append({"pid": pid, "status": status})
        record["reaped_descendants"] = reaped
        record["elapsed_seconds"] = round(time.time() - started, 3)
        (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
