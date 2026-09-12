#!/usr/bin/env python3
"""Probe live OpenAI endpoints for preferred-sampling precedence and protocol paths."""

import argparse
import json
import urllib.request


PREFERRED = {
    "temperature": 0.0,
    "top_p": 0.55,
    "top_k": 1,
    "presence_penalty": 1.25,
    "repetition_penalty": 1.1,
    "seed": 17,
}


def post(base_url, path, payload, stream=False):
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if stream:
            return {
                "status": response.status,
                "lines": [line.decode().strip() for line in response if line.strip()],
            }
        return {"status": response.status, "body": json.load(response)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="tiny-llama")
    parser.add_argument("--expect-chat-precedence", action="store_true")
    args = parser.parse_args()

    chat = {
        "model": args.model,
        "messages": [{"role": "user", "content": "hello world"}],
        "max_tokens": 8,
    }
    completion = {"model": args.model, "prompt": "hello world", "max_tokens": 8}
    results = {
        "chat_omitted": post(args.base_url, "/v1/chat/completions", chat),
        "chat_explicit_preferred": post(
            args.base_url, "/v1/chat/completions", {**chat, **PREFERRED}
        ),
        "chat_explicit_override": post(
            args.base_url,
            "/v1/chat/completions",
            {**chat, "temperature": 1.0, "top_p": 1.0, "top_k": -1,
             "presence_penalty": 0.0, "repetition_penalty": 1.0, "seed": 23},
        ),
        "completion_omitted": post(args.base_url, "/v1/completions", completion),
        "completion_explicit_preferred": post(
            args.base_url, "/v1/completions", {**completion, **PREFERRED}
        ),
        "completion_batch_two": post(
            args.base_url, "/v1/completions",
            {"model": args.model, "prompt": ["hello world", "alpha beta"],
             "max_tokens": 8},
        ),
        "chat_stream": post(
            args.base_url, "/v1/chat/completions", {**chat, "stream": True}, True
        ),
        "completion_stream": post(
            args.base_url, "/v1/completions", {**completion, "stream": True}, True
        ),
    }
    omitted = results["chat_omitted"]["body"]["choices"][0]["message"]["content"]
    explicit = results["chat_explicit_preferred"]["body"]["choices"][0]["message"]["content"]
    results["checks"] = {
        "all_http_200": all(value["status"] == 200 for value in results.values()),
        "batch_has_two_choices": len(
            results["completion_batch_two"]["body"]["choices"]
        ) == 2,
        "streams_terminate": all(
            results[key]["lines"][-1] == "data: [DONE]"
            for key in ("chat_stream", "completion_stream")
        ),
        "omitted_chat_matches_explicit_preferred": omitted == explicit,
    }
    print(json.dumps(results, indent=2, sort_keys=True))
    if args.expect_chat_precedence and not all(results["checks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
