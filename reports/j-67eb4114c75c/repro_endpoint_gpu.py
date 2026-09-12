"""Issue-specific ROCm reproducer for benchmark endpoint process creation.

Run from the repository root with the prepared interpreter. This is a process
and accelerator fixture, not a full model-serving reproduction.
"""

import os
import sys
import time

import torch

from sglang.benchmark import endpoint

MARKER = "/tmp/amdpilot-repo-j-67eb4114c75c/endpoint_gpu_child_ok"


class Args:
    host = "127.0.0.1"
    port = 39999

    def resolve_once(self):
        torch.cuda.init()
        print(
            f"parent pid={os.getpid()} initialized={torch.cuda.is_initialized()}",
            flush=True,
        )


def child_gpu_touch(_args):
    print(
        f"child pid={os.getpid()} initialized={torch.cuda.is_initialized()}",
        flush=True,
    )
    value = torch.zeros(1, device="cuda") + 7
    with open(MARKER, "w") as marker:
        marker.write(f"value={value.cpu().item()}\n")
    time.sleep(60)


def main():
    if os.path.exists(MARKER):
        os.unlink(MARKER)
    endpoint.server_is_up = lambda *args, **kwargs: os.path.exists(MARKER)
    endpoint.DEFAULT_TIMEOUT = 60

    try:
        process, url = endpoint.launch_or_reuse_server(child_gpu_touch, Args())
        print(f"endpoint ready: url={url} child_alive={process.is_alive()}")
        with open(MARKER) as marker:
            print(marker.read().strip())
        process.terminate()
        process.join()
    except Exception as exc:
        print(f"parent observed: {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
