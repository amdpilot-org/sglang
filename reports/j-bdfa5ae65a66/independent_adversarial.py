from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator


def allocator(size=8, need_sort=False, device="cpu"):
    return PagedTokenToKVPoolAllocator(
        size=size,
        page_size=4,
        dtype=torch.float16,
        device=device,
        kvcache=None,
        need_sort=need_sort,
    )


# Multi-request extend demand exceeds capacity by one page.
a = allocator(size=12)
before = a.free_pages.clone()
prefix = torch.tensor([0, 3], dtype=torch.int64)
seq = torch.tensor([8, 12], dtype=torch.int64)  # 2 + 2 pages = 4, only 3 free
last = torch.tensor([-1, 2], dtype=torch.int64)
with patch("sglang.srt.mem_cache.allocator.paged.alloc_extend_kernel") as kernel:
    result = a.alloc_extend(prefix, prefix, seq, seq, last, extend_num_tokens=17)
assert result is None
assert kernel.__getitem__.call_count == 0
assert torch.equal(a.free_pages, before)

# An extend needing zero new pages must still launch to map new token indices.
a = allocator(size=4)
a.free_pages = a.free_pages[:0]
prefix = torch.tensor([1], dtype=torch.int64)
seq = torch.tensor([3], dtype=torch.int64)
last = torch.tensor([0], dtype=torch.int64)
with patch("sglang.srt.mem_cache.allocator.paged.alloc_extend_kernel") as kernel:
    result = a.alloc_extend(prefix, prefix, seq, seq, last, extend_num_tokens=2)
assert result is not None
assert kernel.__getitem__.call_count == 1
assert len(a.free_pages) == 0

# Decode batch can exceed capacity even when batch size itself does not: every
# sequence crosses a page boundary here, while staged pages remain insufficient.
a = allocator(size=16, need_sort=True)
held = a.alloc(12)
a.free(held[:4])  # one staged + one immediately free = two total pages
before_total = a.available_size()
seq = torch.tensor([5, 9, 13], dtype=torch.int64)
last = torch.tensor([3, 7, 11], dtype=torch.int64)
with patch("sglang.srt.mem_cache.allocator.paged.alloc_decode_kernel") as kernel:
    result = a.alloc_decode(seq, seq, last)
assert result is None
assert kernel.__getitem__.call_count == 0
assert a.available_size() == before_total

print("independent_adversarial: PASS")
