from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    DeepSeekV4StateHostPool,
)


def main():
    device = "cuda"
    page_size = 4
    device_pages = 6
    host_pages = 4

    paged_buffers = [
        (torch.arange(device_pages * 13, device=device, dtype=torch.int32) + i * 1000)
        .view(device_pages, 13)
        .view(torch.uint8)
        for i in range(3)
    ]
    paged = DeepSeekV4PagedHostPool(
        pool_name="gpu-check-paged",
        device_buffers=paged_buffers,
        item_bytes=paged_buffers[0].shape[1],
        num_host_pages=host_pages,
        slot_page_size=page_size,
        layout="layer_first",
    )
    host_indices = torch.tensor([4, 5, 6, 7, 12, 13, 14, 15])
    device_indices = torch.tensor([8, 9, 10, 11, 16, 17, 18, 19], device=device)
    paged.backup_from_device_all_layer(
        None, host_indices, device_indices, io_backend="direct"
    )
    torch.cuda.synchronize()
    for layer, (src, dst) in enumerate(zip(paged_buffers, paged.kv_buffer)):
        expected = src[[2, 4]].cpu()
        actual = dst[[1, 3]]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print(f"paged_layer_{layer}_bytes={actual.numel()} exact=True")

    ring_size = 2
    state_pools = []
    for layer in range(2):
        state = (
            torch.arange(device_pages * ring_size * 5, device=device, dtype=torch.int16)
            + layer * 1000
        ).view(device_pages * ring_size, 5)
        state_pools.append(
            SimpleNamespace(
                ring_size=ring_size,
                kv_score_buffer=SimpleNamespace(kv_score=state),
            )
        )
    states = DeepSeekV4StateHostPool(
        pool_name="gpu-check-state",
        state_pools=state_pools,
        num_host_pages=host_pages,
        swa_page_size=page_size,
        layout="layer_first",
    )
    states.backup_from_device_all_layer(
        None, host_indices, device_indices, io_backend="direct"
    )
    torch.cuda.synchronize()
    for layer, (src, dst) in enumerate(zip(states.device_page_views, states.kv_buffer)):
        expected = src[[2, 4]].cpu()
        actual = dst[[1, 3]]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print(f"state_layer_{layer}_bytes={actual.numel()} exact=True")

    print(f"device={torch.cuda.get_device_name(0)} hip={torch.version.hip}")


if __name__ == "__main__":
    main()
