#!/usr/bin/env python3
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = "http://127.0.0.1:31322"
MODEL = "/models/DeepSeek-V4-Flash-0731"
SEED = 12345
MAX_TOKENS = 256
CASES = {
    "arithmetic": {
        "messages": [
            {
                "role": "user",
                "content": "Explain 12 + 34 in two sentences.",
            }
        ],
        "checks": ["contains 46", "at least two sentences"],
    },
    "day_night": {
        "messages": [
            {
                "role": "user",
                "content": "Explain why day and night occur in three sentences.",
            }
        ],
        "checks": ["mentions Earth's rotation", "at least three sentences"],
    },
    "france": {
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France? Answer in a complete sentence, then add one relevant factual sentence.",
            }
        ],
        "checks": ["contains Paris", "at least two sentences"],
    },
    "sum_squares": {
        "messages": [
            {
                "role": "user",
                "content": "Implement a Python function sum_squares(nums) that returns the sum of squares. Verify it returns 0 for [], 14 for [1, 2, 3], and 13 for [-2, 3].",
            }
        ],
        "checks": ["defines sum_squares", "verifies 0, 14, and 13"],
    },
}


def post(path: str, payload: dict) -> tuple[dict, float, int]:
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8")
        status = error.code
    elapsed = time.monotonic() - started
    return json.loads(body), elapsed, status


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
            "stop_token_ids": [1],
        },
    }


def main() -> None:
    encoding = json.loads(Path("/job/raw/dsv4_official_encoding.json").read_text())
    results = {"chat": {}, "raw_generate": {}, "metadata": {
        "base_url": BASE_URL,
        "model": MODEL,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": False,
        "eos_token_id": encoding["eos_token_id"],
    }}
    for case_name, case in CASES.items():
        results["chat"][case_name] = []
        for repeat in range(1, 3):
            response, elapsed, status = post(
                "/v1/chat/completions", chat_payload(case["messages"])
            )
            results["chat"][case_name].append({
                "repeat": repeat,
                "status": status,
                "elapsed_seconds": elapsed,
                "response": response,
            })
    for case_name, case in CASES.items():
        results["raw_generate"][case_name] = []
        prompt = encoding["cases"][case_name]["formatted_prompt"]
        for repeat in range(1, 3):
            response, elapsed, status = post("/generate", raw_payload(prompt))
            results["raw_generate"][case_name].append({
                "repeat": repeat,
                "status": status,
                "elapsed_seconds": elapsed,
                "request_prompt": prompt,
                "response": response,
            })
    output = Path("/job/raw/dsv4_generation_validation.json")
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
