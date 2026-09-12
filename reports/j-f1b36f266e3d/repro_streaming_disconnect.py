#!/usr/bin/env python3
"""Deterministic HTTP reproduction for issue 36475 using token IDs."""

import argparse
import json
import time

import httpx


def stream_turn(client, url, input_ids, session_id, max_new_tokens, stop_after=None):
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        "session_params": {"id": session_id, "rid": None},
        "stream": True,
    }
    chunks = []
    with client.stream("POST", f"{url}/generate", json=payload, timeout=60) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data:") or line.strip() == "data: [DONE]":
                continue
            chunks.append(json.loads(line[5:].strip()))
            if stop_after is not None and len(chunks) >= stop_after:
                break
    return chunks


def summary(chunks):
    last = chunks[-1]
    return {
        "chunks": len(chunks),
        "output_ids": last.get("output_ids"),
        "meta_info": last.get("meta_info", {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("--post-disconnect-delay", type=float, default=0)
    args = parser.parse_args()
    session_id = f"disconnect-race-{time.time_ns()}"
    result = {"session_id": session_id, "post_disconnect_delay": args.post_disconnect_delay}
    with httpx.Client(trust_env=False) as client:
        opened = client.post(
            f"{args.url}/open_session",
            json={"capacity_of_str_len": 10000, "session_id": session_id, "streaming": True},
        )
        opened.raise_for_status()
        turn0 = stream_turn(client, args.url, [4] * 32, session_id, 24)
        result["turn0"] = summary(turn0)
        turn1 = stream_turn(client, args.url, [5] * 16, session_id, 96, stop_after=1)
        result["turn1_disconnected"] = summary(turn1)
        if args.post_disconnect_delay:
            time.sleep(args.post_disconnect_delay)
        turn2 = stream_turn(client, args.url, [6] * 16, session_id, 16)
        result["turn2"] = summary(turn2)
        result["expected_turn2_prompt_tokens"] = 32 + 24 + 16
        result["turn2_visible_busy"] = result["turn2"]["meta_info"].get(
            "finish_reason", {}
        ).get("message") == "Streaming session already has an active request."
        result["turn2_prompt_ok"] = (
            result["turn2"]["meta_info"].get("prompt_tokens")
            == result["expected_turn2_prompt_tokens"]
        )
        # A second immediate retry is the ownership regression: the first
        # rejected retry must not clear the interrupted request's inflight flag.
        turn3 = stream_turn(client, args.url, [7] * 16, session_id, 16)
        result["turn3_immediate"] = summary(turn3)
        result["turn3_visible_busy"] = result["turn3_immediate"]["meta_info"].get(
            "finish_reason", {}
        ).get("message") == "Streaming session already has an active request."
        # Let the HTTP disconnect background abort settle, then verify rollback
        # to turn 0 plus this new input.
        time.sleep(2.5)
        turn4 = stream_turn(client, args.url, [8] * 16, session_id, 16)
        result["turn4_after_abort"] = summary(turn4)
        result["turn4_prompt_ok"] = (
            result["turn4_after_abort"]["meta_info"].get("prompt_tokens")
            == result["expected_turn2_prompt_tokens"]
        )
        time.sleep(4)
        try:
            result["health_after_wait"] = client.get(f"{args.url}/health", timeout=5).status_code
        except Exception as exc:
            result["health_after_wait"] = f"{type(exc).__name__}: {exc}"
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    immediate_retry_safe = result["turn2_prompt_ok"] or (
        result["turn2_visible_busy"] and result["turn3_visible_busy"]
    )
    return 0 if immediate_retry_safe and result["health_after_wait"] == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
