"""Single-GPU algebra check only; not a distributed transport qualification."""

import json

import torch


def main():
    torch.manual_seed(36820)
    device = torch.device("cuda")
    ranks, tokens, hidden = 4, 7, 32
    active = torch.tensor([True, False, True, False], device=device)

    # Model per-rank post-expert partials. Inactive ranks are provably zero.
    partials = torch.randn(ranks, tokens, hidden, device=device, dtype=torch.float32)
    partials[~active] = 0
    collective = partials.sum(dim=0)
    sparse_combine = partials[active].sum(dim=0)

    # Independent high-precision CPU reference, not another GPU reduction.
    cpu_reference = partials.cpu().double().numpy().sum(axis=0)
    max_gpu_path_diff = (collective - sparse_combine).abs().max().item()
    max_reference_diff = (
        collective.cpu().double() - torch.from_numpy(cpu_reference)
    ).abs().max().item()

    result = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "active_ranks": active.cpu().tolist(),
        "collective_payload_elements": ranks * tokens * hidden,
        "ideal_sparse_payload_elements": int(active.sum().item()) * tokens * hidden,
        "max_gpu_path_diff": max_gpu_path_diff,
        "max_cpu_float64_reference_diff": max_reference_diff,
        "scope": "local combine algebra only; no multi-rank transport executed",
    }
    print(json.dumps(result, indent=2))
    assert max_gpu_path_diff == 0.0
    assert max_reference_diff < 1e-5


if __name__ == "__main__":
    main()
