#!/usr/bin/env python3
import argparse
import json
import time
import urllib.request
from pathlib import Path


def get(url: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return {"status": response.status, "body": json.loads(response.read().decode("utf-8"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--timeout", type=int, default=1500)
    args = parser.parse_args()

    started = time.time()
    deadline = started + args.timeout
    last_error = None
    while time.time() < deadline:
        try:
            health = get("http://127.0.0.1:31322/health", timeout=5)
            if health["status"] == 200:
                model_info = get("http://127.0.0.1:31322/get_model_info")
                server_info = get("http://127.0.0.1:31322/get_server_info")
                result = {
                    "arm": args.arm,
                    "started_epoch": started,
                    "ready_epoch": time.time(),
                    "elapsed_seconds": time.time() - started,
                    "health": health,
                    "model_info": model_info,
                    "server_info": server_info,
                }
                output = Path(f"/job/evidence/{args.arm}/server-ready.json")
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return
        except Exception as error:
            last_error = repr(error)
        time.sleep(5)
    raise SystemExit(f"server timeout after {args.timeout}s: {last_error}")


if __name__ == "__main__":
    main()
