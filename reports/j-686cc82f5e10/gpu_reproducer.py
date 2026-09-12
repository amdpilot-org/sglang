"""Focused gfx950 evidence for paged allocator success and OOM paths."""

from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator import paged


def make_allocator():
    return paged.PagedTokenToKVPoolAllocator(
        size=8,
        page_size=4,
        dtype=torch.float16,
        device="cuda",
        kvcache=None,
        need_sort=False,
    )


def main():
    print("device=", torch.cuda.get_device_name(0))
    print("capability=", torch.cuda.get_device_capability(0))
    print("torch=", torch.__version__, "hip=", torch.version.hip)

    allocator = make_allocator()
    prefix_lens = torch.tensor([0], dtype=torch.int64, device="cuda")
    seq_lens = torch.tensor([8], dtype=torch.int64, device="cuda")
    last_loc = torch.tensor([-1], dtype=torch.int64, device="cuda")
    actual = allocator.alloc_extend(
        prefix_lens,
        prefix_lens.cpu(),
        seq_lens,
        seq_lens.cpu(),
        last_loc,
        extend_num_tokens=8,
    )
    torch.cuda.synchronize()
    expected = torch.arange(4, 12, dtype=torch.int64)
    print("extend_actual=", actual.cpu().tolist())
    print("extend_expected=", expected.tolist())
    torch.testing.assert_close(actual.cpu(), expected)

    allocator = make_allocator()
    seq_lens = torch.tensor([5, 9], dtype=torch.int64, device="cuda")
    last_loc = torch.tensor([3, 7], dtype=torch.int64, device="cuda")
    actual = allocator.alloc_decode(seq_lens, seq_lens.cpu(), last_loc)
    torch.cuda.synchronize()
    expected = torch.tensor([4, 8], dtype=torch.int64)
    print("decode_actual=", actual.cpu().tolist())
    print("decode_expected=", expected.tolist())
    torch.testing.assert_close(actual.cpu(), expected)

    allocator = make_allocator()
    with patch.object(
        paged, "alloc_extend_kernel", wraps=paged.alloc_extend_kernel
    ) as kernel:
        result = allocator.alloc_extend(
            torch.tensor([0], dtype=torch.int64, device="cuda"),
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([12], dtype=torch.int64, device="cuda"),
            torch.tensor([12], dtype=torch.int64),
            torch.tensor([-1], dtype=torch.int64, device="cuda"),
            extend_num_tokens=12,
        )
        print(
            "extend_oom=",
            result is None,
            "kernel_launches=",
            kernel.__getitem__.call_count,
        )
        assert result is None and kernel.__getitem__.call_count == 0

    allocator = make_allocator()
    with patch.object(
        paged, "alloc_decode_kernel", wraps=paged.alloc_decode_kernel
    ) as kernel:
        result = allocator.alloc_decode(
            torch.tensor([5, 9, 13], dtype=torch.int64, device="cuda"),
            torch.tensor([5, 9, 13], dtype=torch.int64),
            torch.tensor([3, 7, 11], dtype=torch.int64, device="cuda"),
        )
        print(
            "decode_oom=",
            result is None,
            "kernel_launches=",
            kernel.__getitem__.call_count,
        )
        assert result is None and kernel.__getitem__.call_count == 0


if __name__ == "__main__":
    main()
