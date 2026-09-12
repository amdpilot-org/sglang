"""One-GPU numerical check for draft-only EPLB padding and remapping."""

import numpy as np
import torch

from sglang.srt.eplb.expert_location import resolve_draft_redundant_experts


def main() -> None:
    torch.manual_seed(7893)
    rng = np.random.default_rng(7893)
    num_logical = 256
    ep_size = 72
    redundancy = resolve_draft_redundant_experts(num_logical, 0, ep_size)
    num_physical = num_logical + redundancy

    # A trivial EPLB layout duplicates the first logical experts into padding.
    physical_to_logical = np.arange(num_physical, dtype=np.int64) % num_logical
    logical_values = rng.standard_normal((num_logical, 8), dtype=np.float32)
    physical_values = logical_values[physical_to_logical]
    logical_ids = rng.integers(0, num_logical, size=4096, dtype=np.int64)
    replica_ordinal = rng.integers(0, 2, size=logical_ids.shape, dtype=np.int64)
    physical_ids = logical_ids.copy()
    duplicate_mask = (logical_ids < redundancy) & (replica_ordinal == 1)
    physical_ids[duplicate_mask] += num_logical

    gpu_values = torch.from_numpy(physical_values).cuda()
    gpu_ids = torch.from_numpy(physical_ids).cuda()
    actual = gpu_values[gpu_ids].cpu().numpy()
    expected = logical_values[logical_ids]
    np.testing.assert_array_equal(actual, expected)
    assert num_physical % ep_size == 0
    print(
        f"PASS device={torch.cuda.get_device_name(0)} logical={num_logical} "
        f"physical={num_physical} ep_size={ep_size} samples={logical_ids.size} "
        "exact_match=true"
    )


if __name__ == "__main__":
    main()
