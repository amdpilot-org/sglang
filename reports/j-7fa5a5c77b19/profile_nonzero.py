"""Reproduce the call-site placement, overhead, and GPU-sync audit."""

import statistics
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.configs.janus_pro import VLChatProcessor
from sglang.srt.models.phi4mm_utils import adaptive_enc_mask


class _LogNonzero(TorchDispatchMode):
    def __init__(self):
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.nonzero.default:
            tensor = args[0]
            self.calls.append(
                (str(tensor.device), tuple(tensor.shape), str(tensor.dtype))
            )
        return func(*args, **(kwargs or {}))


class _Tokenizer:
    def encode(self, prompt):
        return [11, 99, 12, 99, 13]


class _ImageProcessor:
    def __call__(self, images, return_tensors):
        return SimpleNamespace(pixel_values=torch.empty((0, 3, 1, 1)))


def _median_us(function, iterations, rounds=11):
    samples = []
    for _ in range(rounds):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        samples.append((time.perf_counter_ns() - start) / iterations / 1_000)
    return statistics.median(samples)


def _legacy_adaptive_enc_mask(x_len, starts, left_window=0, right_window=0):
    starts = torch.Tensor(starts).long()
    start_pad = torch.nn.functional.pad(starts, (1, 0))
    end_pad = torch.nn.functional.pad(starts, (0, 1), value=x_len)
    seq_range = torch.arange(x_len).unsqueeze(-1)
    idx = ((seq_range < end_pad) & (seq_range >= start_pad)).nonzero()[:, 1]
    grid = torch.arange(x_len).unsqueeze(0).expand(x_len, -1)
    idx_left = (idx - left_window).clamp_min(0)
    idx_right = (idx + right_window).clamp_max(len(starts))
    return (grid >= start_pad[idx_left].unsqueeze(-1)) & (
        grid < end_pad[idx_right].unsqueeze(-1)
    )


def main():
    fake = SimpleNamespace(
        tokenizer=_Tokenizer(),
        image_processor=_ImageProcessor(),
        image_id=99,
        image_start_id=101,
        image_end_id=102,
        num_image_tokens=4,
        add_special_token=False,
    )
    fake.add_image_token = lambda **kwargs: VLChatProcessor.add_image_token(
        fake, **kwargs
    )
    with _LogNonzero() as log:
        VLChatProcessor.process_one(fake, prompt="x", images=[])
    print("janus_nonzero", log.calls)

    for x_len in (18, 64, 256, 1024, 4096):
        starts = np.arange(0, x_len, 18)
        iterations = 1_000 if x_len <= 256 else (100 if x_len <= 1024 else 10)
        expected = _legacy_adaptive_enc_mask(x_len, starts, 6)
        actual = adaptive_enc_mask(x_len, starts, 6)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        old = _median_us(
            lambda: _legacy_adaptive_enc_mask(x_len, starts, 6), iterations
        )
        new = _median_us(lambda: adaptive_enc_mask(x_len, starts, 6), iterations)
        print("phi_us", x_len, old, new)

    if not torch.cuda.is_available():
        print("gpu_sync skipped: no assigned GPU")
        return

    values = torch.arange(4096, device="cuda")
    boundaries = torch.tensor([0, 1024, 2048], device="cuda")
    operations = {
        "nonzero": lambda: (values % 2 == 0).nonzero(),
        "nonzero_tuple": lambda: (values % 2 == 0).nonzero(as_tuple=True)[0],
        "bucketize": lambda: torch.bucketize(values, boundaries, right=True),
        "add": lambda: values + 1,
    }
    cycles = 100_000_000
    for name, operation in operations.items():
        operation()
        torch.cuda.synchronize()
        samples = []
        for _ in range(11):
            torch.cuda.synchronize()
            torch.cuda._sleep(cycles)
            start = time.perf_counter_ns()
            operation()
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
            torch.cuda.synchronize()
        print("gpu_pending_work_ms", name, statistics.median(samples), samples)


if __name__ == "__main__":
    main()
