import inspect

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNAttnBackend,
    _causal_conv1d_with_cache_dtype,
)


def reference(x, weight, bias, state):
    window = torch.cat((state, x), dim=-1)
    output = F.conv1d(
        window.unsqueeze(0),
        weight.unsqueeze(1),
        bias,
        groups=x.shape[0],
    )
    return F.silu(output).squeeze(0), window[:, -state.shape[-1] :]


print("module", inspect.getsourcefile(GDNAttnBackend))
print("device", torch.cuda.get_device_name(0), "hip", torch.version.hip)
mis_source = inspect.getsource(GDNAttnBackend._forward_mis_segments)
assert "mixed_qkv = _causal_conv1d_with_cache_dtype(" in mis_source
print("mis_uses_dtype_helper", True)

device = "cuda"
dim, width, tokens = 256, 4, 8
for cache_dtype in (torch.float32, torch.float16):
    torch.manual_seed(31719)
    x = torch.randn(dim, tokens, device=device, dtype=torch.bfloat16)
    weight = torch.randn(dim, width, device=device, dtype=torch.bfloat16)
    bias = torch.randn(dim, device=device, dtype=torch.bfloat16)
    states = torch.randn(2, dim, width - 1, device=device, dtype=cache_dtype)
    before = states.clone()
    expected_output, expected_state = reference(
        x, weight, bias, before[0].to(torch.bfloat16)
    )
    output = _causal_conv1d_with_cache_dtype(
        x,
        weight,
        bias,
        activation="silu",
        conv_states=states,
        has_initial_state=torch.tensor([True, True], device=device),
        cache_indices=torch.tensor([0, -1], device=device, dtype=torch.int32),
        query_start_loc=torch.tensor(
            [0, tokens, tokens], device=device, dtype=torch.int32
        ),
        seq_lens_cpu=[tokens, 0],
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected_output, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(
        states[0], expected_state.to(cache_dtype), atol=5e-2, rtol=1e-2
    )
    torch.testing.assert_close(states[-1], before[-1], atol=0, rtol=0)
    print(
        "SUCCESS",
        "activation=torch.bfloat16",
        f"cache={cache_dtype}",
        "output_matches_grouped_conv=True",
        "active_state_matches_reference=True",
        "pad_row_unchanged=True",
    )

print("CORRECTED_BOUNDARIES_PASS")
