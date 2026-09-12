"""Independent gfx950 check for page-exact PagedTokenToKVPoolAllocator frees."""

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator


PAGE_SIZE = 4
allocator = PagedTokenToKVPoolAllocator(
    size=64,
    page_size=PAGE_SIZE,
    dtype=torch.float16,
    device="cuda",
    kvcache=None,
    need_sort=False,
)

assert torch.cuda.device_count() == 1
row = allocator.alloc(4 * PAGE_SIZE)
cases = ((0, 1), (PAGE_SIZE, PAGE_SIZE + 3), (2 * PAGE_SIZE, 4 * PAGE_SIZE - 1))
for start, end in cases:
    expected = torch.unique(row[start:end] // PAGE_SIZE).sort().values
    before = allocator.free_pages.numel()
    allocator.free_segment(row[start:end], start_pos=start)
    count = allocator.free_pages.numel() - before
    actual = allocator.free_pages[:count].sort().values
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(f"case={start}:{end} expected={expected.tolist()} actual={actual.tolist()}")

props = torch.cuda.get_device_properties(0)
print(f"device={props.name} arch={props.gcnArchName} cases={len(cases)} status=pass")
