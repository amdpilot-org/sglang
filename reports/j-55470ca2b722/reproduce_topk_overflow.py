import json

import torch
from sgl_kernel import fast_topk_v2


def run_case(batch: int, length: int, seed: int) -> dict:
    k = 2048
    torch.manual_seed(seed)
    score = torch.randn(batch, length, device="cuda", dtype=torch.float32)
    lengths = torch.full((batch,), length, device="cuda", dtype=torch.int32)
    indices = fast_topk_v2(score, lengths, k)
    torch.cuda.synchronize()

    selected = score.gather(1, indices.long()).sort(dim=1, descending=True).values
    reference = torch.topk(score, k, dim=1, sorted=True).values
    wrong = (selected != reference).any(dim=1)

    bits = score.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF
    keys = torch.where(bits & 0x8000 != 0, bits ^ 0xFFFF, bits | 0x8000)
    bins = (keys >> 8) & 0xFF
    kth_indices = torch.topk(score, k, dim=1).indices[:, -1:]
    kth_bins = bins.gather(1, kth_indices)
    bucket_populations = (bins == kth_bins).sum(dim=1)

    sorted_indices = indices.long().sort(dim=1).values
    duplicates = (sorted_indices[:, 1:] == sorted_indices[:, :-1]).any()
    return {
        "batch": batch,
        "length": length,
        "seed": seed,
        "wrong_rows": int(wrong.sum()),
        "bucket_population_min": int(bucket_populations.min()),
        "bucket_population_median": int(bucket_populations.median()),
        "bucket_population_max": int(bucket_populations.max()),
        "indices_in_range": bool((indices >= 0).all() and (indices < length).all()),
        "duplicate_indices": bool(duplicates),
    }


if __name__ == "__main__":
    device = torch.cuda.get_device_properties(0)
    result = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device_name": device.name,
        "architecture": device.gcnArchName,
        "cases": [
            run_case(batch=64, length=65536, seed=0),
            run_case(batch=64, length=256 * 1024, seed=0),
            run_case(batch=4, length=1024 * 1024, seed=1),
        ],
    }
    print(json.dumps(result, indent=2))
