import sys

import pytest
import sgl_kernel
import torch
from sgl_kernel.elementwise import copy_to_gpu_no_ce


@pytest.mark.parametrize("size", [16, 17, 32, 64, 72, 512, 513, 1024, 1025])
def test_copy_to_gpu_no_ce(size):
    """Copies both specialized and fallback local-expert vector sizes."""
    tensor_cpu = torch.randint(0, 1000000, (size,), dtype=torch.int32, device="cpu")
    tensor_gpu = torch.empty_like(tensor_cpu, device="cuda")
    copy_to_gpu_no_ce(tensor_cpu, tensor_gpu)
    assert torch.all(tensor_cpu.cuda() == tensor_gpu)


def test_copy_to_gpu_no_ce_rejects_empty_input():
    """Rejects an empty vector, which cannot produce a kernel launch."""
    size = 0
    tensor_cpu = torch.empty(size, dtype=torch.int32, device="cpu")
    tensor_gpu = torch.empty_like(tensor_cpu, device="cuda")

    with pytest.raises(RuntimeError, match="does not support empty tensors"):
        copy_to_gpu_no_ce(tensor_cpu, tensor_gpu)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
