"""Allocate an AITER-sized workspace on the assigned GPU and verify its extent."""

import torch

from sglang.srt.layers.attention.aiter_backend import (
    get_aiter_workspace_size_bytes,
)


def main():
    props = torch.cuda.get_device_properties(0)
    kwargs = dict(
        max_num_reqs=123,
        num_heads=32,
        max_context_len=8192,
        head_dim=128,
    )
    expected = 123 * 32 * 32 * (128 * 4 + 2 * 4)
    calculated = get_aiter_workspace_size_bytes(**kwargs)
    workspace = torch.empty(calculated, dtype=torch.uint8, device="cuda:0")
    workspace[0] = 17
    workspace[-1] = 29
    torch.cuda.synchronize()

    print(f"device={props.name}")
    print(f"gcn_arch_name={props.gcnArchName}")
    print(f"calculated_bytes={calculated}")
    print(f"allocated_bytes={workspace.numel() * workspace.element_size()}")
    print(f"boundary_values={workspace[0].item()},{workspace[-1].item()}")
    assert calculated == expected
    assert workspace.numel() * workspace.element_size() == expected
    assert (workspace[0].item(), workspace[-1].item()) == (17, 29)


if __name__ == "__main__":
    main()
