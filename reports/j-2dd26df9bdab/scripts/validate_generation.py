#!/usr/bin/env python3
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = "http://127.0.0.1:31322"
MODEL = "/models/DeepSeek-V4-Flash-0731"
SEED = 12345
MAX_TOKENS = 256
EOS_TOKEN_ID = 1


def post(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8")
        status = error.code
    elapsed = time.monotonic() - started
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = None
    return {
        "status": status,
        "elapsed_seconds": elapsed,
        "response": parsed,
        "response_raw": body,
    }


def chat_payload(messages: list[dict]) -> dict:
    return {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
        "ignore_eos": False,
        "return_token_ids": True,
        "chat_template_kwargs": {"thinking": False},
    }


def raw_payload(prompt: str) -> dict:
    return {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": MAX_TOKENS,
            "sampling_seed": SEED,
            "ignore_eos": False,
            "stop_token_ids": [EOS_TOKEN_ID],
        },
    }


def durable_write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--header-sha256", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--encoding", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    encoding = json.loads(Path(args.encoding).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pid_path = Path(f"/job/evidence/{args.arm}/server.pid")
    server_pid = pid_path.read_text().strip() if pid_path.exists() else None
    metadata = {
        "arm": args.arm,
        "source": args.source,
        "header_sha256": args.header_sha256,
        "cache_root": args.cache_root,
        "server_pid": server_pid,
        "base_url": BASE_URL,
        "model": MODEL,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": False,
        "eos_token_id": EOS_TOKEN_ID,
        "encoding_dsv4_file": encoding["encoding_dsv4_file"],
        "bos_token_id": encoding["bos_token_id"],
        "thinking_mode": encoding["thinking_mode"],
        "reasoning_effort_profile": encoding["reasoning_effort_profile"],
    }

    records = []
    for mode in ("chat", "raw_generate"):
        for case_name, case in encoding["cases"].items():
            for repeat in (1, 2):
                if mode == "chat":
                    request_payload = chat_payload(case["messages"])
                    endpoint = "/v1/chat/completions"
                    prompt = case["formatted_prompt"]
                    prompt_token_ids = case["prompt_token_ids"]
                else:
                    request_payload = raw_payload(case["formatted_prompt"])
                    endpoint = "/generate"
                    prompt = case["formatted_prompt"]
                    prompt_token_ids = case["prompt_token_ids"]
                response_record = post(endpoint, request_payload)
                record = {
                    "metadata": metadata,
                    "mode": mode,
                    "case": case_name,
                    "repeat": repeat,
                    "endpoint": endpoint,
                    "request": request_payload,
                    "formatted_prompt": prompt,
                    "prompt_token_ids": prompt_token_ids,
                    **response_record,
                }
                records.append(record)
                append_jsonl(output_dir / "requests.jsonl", record)
                durable_write(
                    output_dir
                    / f"{mode}-{case_name}-repeat-{repeat}.json",
                    record,
                )

    final = {"metadata": metadata, "records": records}
    durable_write(output_dir / "generation-results.json", final)
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
