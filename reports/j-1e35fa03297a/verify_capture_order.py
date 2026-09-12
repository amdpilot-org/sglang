"""Focused gfx950 check for the eager-warmup-before-graph ordering.

This does not emulate DeepEP or the reported distributed Kimi workload.  It
checks the relevant CUDA/HIP graph boundary with a lazy allocation and a
numerical replay result on the assigned GPU.
"""

import torch

from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)


class _Group:
    def barrier(self) -> None:
        pass


class _ModelRunner:
    tp_group = _Group()


class _Runner:
    device_module = torch.cuda
    model_runner = _ModelRunner()
    enable_profile_cuda_graph = False


def main() -> None:
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    allocations = 0
    workspace = None
    capture_state = []

    x = torch.arange(32, device=device, dtype=torch.float32)

    def forward():
        nonlocal allocations, workspace
        capture_state.append(torch.cuda.is_current_stream_capturing())
        if workspace is None:
            workspace = torch.empty_like(x)
            allocations += 1
        torch.mul(x, 3.0, out=workspace)
        return workspace

    backend = FullCudaGraphBackend(_Runner())
    stream = torch.cuda.Stream()
    with backend.capture_session(stream):
        backend.capture_one("focused-order-check", forward)
    output = backend._outputs["focused-order-check"]

    x.copy_(torch.arange(32, device=device, dtype=torch.float32) + 7)
    backend._graphs["focused-order-check"].replay()
    torch.cuda.synchronize()

    expected = (torch.arange(32, dtype=torch.float32) + 7) * 3.0
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert allocations == 1
    assert capture_state == [False, False, True]

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"torch={torch.__version__} hip={torch.version.hip}")
    print(f"capture_state={capture_state}")
    print(f"lazy_allocations={allocations}")
    print(f"max_abs_error={(output.cpu() - expected).abs().max().item()}")


if __name__ == "__main__":
    main()
