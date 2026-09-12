import traceback

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.linear.gdn_backend import (
    _causal_conv1d_with_cache_dtype,
    _store_tracked_conv_states,
)


def reference(x, weight, bias, states, item_lens, has_initial_state):
    outputs = []
    final_states = []
    for item, state, use_state in zip(
        torch.split(x, item_lens, dim=-1), states, has_initial_state
    ):
        if use_state:
            window = torch.cat((state, item), dim=-1)
            output = F.conv1d(
                window.unsqueeze(0),
                weight.unsqueeze(1),
                bias,
                groups=x.shape[0],
            )
        else:
            window = item
            output = F.conv1d(
                item.unsqueeze(0),
                weight.unsqueeze(1),
                bias,
                padding=weight.shape[-1] - 1,
                groups=x.shape[0],
            )[..., : item.shape[-1]]
        outputs.append(F.silu(output).squeeze(0))
        final_states.append(F.pad(window, (weight.shape[-1] - 1 - window.shape[-1], 0)))
    return torch.cat(outputs, dim=-1), torch.stack(final_states)


def run_case(cache_dtype: torch.dtype, activation_dtype: torch.dtype) -> None:
    device = "cuda"
    dim, width = 256, 4
    item_lens = [8]
    total_tokens = sum(item_lens)
    cache_indices = torch.arange(len(item_lens), device=device, dtype=torch.int32)
    query_start_loc = torch.tensor([0, total_tokens], device=device, dtype=torch.int32)
    has_initial_state = torch.tensor([True], device=device)

    torch.manual_seed(11)
    x = torch.randn(dim, total_tokens, device=device, dtype=activation_dtype)
    weight = torch.randn(dim, width, device=device, dtype=activation_dtype)
    bias = torch.randn(dim, device=device, dtype=activation_dtype)
    conv_states = torch.randn(
        len(item_lens), dim, width - 1, device=device, dtype=cache_dtype
    )

    tracked = x[:, : width - 1].unsqueeze(0)
    _store_tracked_conv_states(conv_states, cache_indices[:1], tracked)
    expected_output, expected_states = reference(
        x,
        weight,
        bias,
        conv_states.to(activation_dtype),
        item_lens,
        has_initial_state,
    )
    output = _causal_conv1d_with_cache_dtype(
        x,
        weight,
        bias,
        conv_states=conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
        seq_lens_cpu=item_lens,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected_output, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(
        conv_states,
        expected_states.to(cache_dtype),
        atol=5e-2,
        rtol=1e-2,
    )
    print(
        "SUCCESS",
        f"activation={activation_dtype}",
        f"cache={cache_dtype}",
        f"output={output.dtype}",
        f"stored={conv_states.dtype}",
    )


print(torch.cuda.get_device_name(0), torch.version.hip)
failed = False
for cache_dtype, activation_dtype in (
    (torch.bfloat16, torch.float16),
    (torch.float32, torch.bfloat16),
    (torch.float16, torch.bfloat16),
    (torch.bfloat16, torch.bfloat16),
):
    try:
        run_case(cache_dtype, activation_dtype)
    except Exception:
        failed = True
        print("FAIL", f"activation={activation_dtype}", f"cache={cache_dtype}")
        traceback.print_exc()

raise SystemExit(1 if failed else 0)
