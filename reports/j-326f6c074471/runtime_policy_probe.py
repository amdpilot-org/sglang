#!/usr/bin/env python3
"""Exercise runtime policy switching through the real HTTP control path."""

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("output")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    def get(path):
        response = session.get(args.base_url + path, timeout=30)
        response.raise_for_status()
        return response.json()

    def set_policy(policy):
        response = requests.post(
            args.base_url + "/set_internal_state",
            json={"server_args": {"schedule_policy": policy}},
            timeout=30,
        )
        return {"policy": policy, "status": response.status_code, "body": response.json()}

    before = get("/server_info")
    generation_payload = {
        "text": "hello world tiny llama rocm gpu test one two three four five six seven eight nine",
        "sampling_params": {"temperature": 0, "max_new_tokens": 96, "ignore_eos": True},
    }
    warm = session.post(
        args.base_url + "/generate", json=generation_payload, timeout=120
    )
    warm.raise_for_status()
    accepted = [set_policy(p) for p in ("lpm", "dfs-weight", "hrrn", "fcfs", "lof", "random", "routing-key")]
    rejected = set_policy("not-a-policy")

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        generation = pool.submit(
            requests.post, args.base_url + "/generate", json=generation_payload, timeout=120
        )
        time.sleep(0.05)
        update_futures = [pool.submit(set_policy, p) for p in ("fcfs", "lpm", "lof", "fcfs")]
        concurrent_updates = [future.result() for future in update_futures]
        generation_response = generation.result()

    after = get("/server_info")
    result = {
        "before": {"schedule_policy": before["schedule_policy"], "startup_time": before["startup_time"]},
        "warm_generation": {"status": warm.status_code, "body": warm.json()},
        "accepted": accepted,
        "rejected": rejected,
        "concurrent_updates": concurrent_updates,
        "generation": {"status": generation_response.status_code, "body": generation_response.json()},
        "after": {"schedule_policy": after["schedule_policy"], "startup_time": after["startup_time"]},
    }
    (output / "http-probe.json").write_text(json.dumps(result, indent=2) + "\n")
    assert all(item["body"] == [True] for item in accepted)
    assert rejected["body"] == [False]
    assert all(item["body"] == [True] for item in concurrent_updates)
    assert generation_response.status_code == 200
    assert generation_response.json()["meta_info"]["cached_tokens"] > 0
    assert before["startup_time"] == after["startup_time"]
    assert after["schedule_policy"] in {item["policy"] for item in concurrent_updates}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
