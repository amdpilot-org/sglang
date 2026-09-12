"""Adversarial reproducer for per-layer KVPress selection in SGLang.

SGLang addresses every attention layer through one request-level slot map.  A
press may nevertheless select a different token set for each layer.  This
script demonstrates that publishing the final layer's selection as the shared
map makes another layer attend to the wrong keys and values.
"""

import torch


def attention(query, keys, values):
    weights = torch.softmax(query @ keys.T, dim=-1)
    return weights @ values


def run_reproducer():
    if not torch.cuda.is_available():
        raise RuntimeError("This reproducer requires the assigned GPU")

    device = torch.device("cuda:0")
    dtype = torch.float32
    query = torch.tensor([[0.7, -0.2]], device=device, dtype=dtype)

    layer0_keys = torch.tensor(
        [[2.0, 0.0], [0.0, 3.0], [1.0, 1.0], [-2.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    layer0_values = torch.tensor(
        [[10.0, 1.0], [20.0, 2.0], [30.0, 3.0], [40.0, 4.0]],
        device=device,
        dtype=dtype,
    )

    # Valid per-layer press choices can differ. A single req_to_token row
    # cannot encode both selections.
    layer0_selection = torch.tensor([0, 2], device=device)
    layer1_selection = torch.tensor([1, 3], device=device)

    reference = attention(
        query, layer0_keys[layer0_selection], layer0_values[layer0_selection]
    )
    shared_map_result = attention(
        query, layer0_keys[layer1_selection], layer0_values[layer1_selection]
    )

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"layer0_reference={reference.cpu().tolist()}")
    print(f"layer0_via_last_layer_shared_map={shared_map_result.cpu().tolist()}")
    print(f"max_abs_error={(reference - shared_map_result).abs().max().item()}")
    assert not torch.allclose(reference, shared_map_result, rtol=1e-5, atol=1e-6)


def test_per_layer_selection_cannot_use_one_shared_map():
    run_reproducer()


if __name__ == "__main__":
    run_reproducer()
