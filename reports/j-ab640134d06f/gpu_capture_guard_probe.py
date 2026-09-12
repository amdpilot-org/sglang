#!/usr/bin/env python3
"""Exercise the capture rejection on the assigned GPU without a communicator."""

import torch

from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    SingleStreamGuard,
)

device = torch.device("cuda:0")
guard = SingleStreamGuard(device)
stream = torch.cuda.Stream(device=device)
graph = torch.cuda.CUDAGraph()

with torch.cuda.stream(stream):
    stream.synchronize()
    try:
        with torch.cuda.graph(graph, stream=stream):
            guard.maybe_serialize()
    except RuntimeError as error:
        message = str(error)
        assert "concurrent graph replay" in message, message
        print(f"device={torch.cuda.get_device_name(device)}")
        print(f"capture_rejected={message}")
    else:
        raise AssertionError("custom all-reduce guard accepted graph capture")

assert guard._last_stream is None
assert guard._last_raw_stream is None
print("guard_state_unchanged=true")
