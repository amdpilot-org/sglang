#!/usr/bin/env python3
"""Exercise the guard from two host threads on two real GPU streams."""

import threading

import torch

from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    SingleStreamGuard,
)

device = torch.device("cuda:0")
guard = SingleStreamGuard(device)
first_stream = torch.cuda.Stream(device=device)
second_stream = torch.cuda.Stream(device=device)
value = torch.zeros((), dtype=torch.int64, device=device)
first_locked = threading.Event()
release_first = threading.Event()


def first():
    with torch.cuda.stream(first_stream), guard.serialize():
        first_locked.set()
        assert release_first.wait(timeout=5)
        value.fill_(1)


def second():
    assert first_locked.wait(timeout=5)
    with torch.cuda.stream(second_stream), guard.serialize():
        value.mul_(2)


t1 = threading.Thread(target=first)
t2 = threading.Thread(target=second)
t1.start()
t2.start()
assert first_locked.wait(timeout=5)
release_first.set()
t1.join(timeout=5)
t2.join(timeout=5)
assert not t1.is_alive() and not t2.is_alive()
torch.cuda.synchronize(device)
actual = value.item()
assert actual == 2, actual
print(f"device={torch.cuda.get_device_name(device)}")
print(f"arch={torch.cuda.get_device_properties(device).gcnArchName}")
print(f"ordered_numeric_result={actual}")
print("cpu_reference=2")
