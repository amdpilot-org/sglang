"""GPU check for DFLASH/DSPARK verify-time additive adjustments."""

from types import SimpleNamespace

import torch

from sglang.srt.speculative.dflash_utils import apply_dflash_verify_logits_adjustments


class SamplingInfo(SimpleNamespace):
    def __len__(self):
        return self.batch_size


device = torch.device("cuda")
properties = torch.cuda.get_device_properties(0)
print(f"device={properties.name}")
print(f"gcn_arch_name={properties.gcnArchName}")

penalties_cpu = torch.tensor(
    [[0.0, float("-inf"), 2.5, -3.0], [1.25, 0.0, -4.0, float("-inf")]],
    dtype=torch.float32,
)
bias_cpu = torch.tensor(
    [[0.5, 1.0, -0.5, 2.0], [-0.25, 3.0, 0.5, 4.0]], dtype=torch.float32
)
draft = 4
logits = torch.zeros((2 * draft, 4), dtype=torch.float16, device=device)
sampling_info = SamplingInfo(
    batch_size=2,
    has_custom_logit_processor=False,
    acc_additive_penalties=penalties_cpu.to(device),
    penalizer_orchestrator=None,
    grammar_mask=None,
    logit_bias=bias_cpu.to(device),
)
apply_dflash_verify_logits_adjustments(
    next_token_logits=logits,
    sampling_info=sampling_info,
    draft_token_num=draft,
)
actual = logits.cpu()
expected = (
    (penalties_cpu + bias_cpu)
    .to(torch.float16)[:, None, :]
    .expand(-1, draft, -1)
    .reshape_as(actual)
)
torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
finite = torch.isfinite(expected)
print(f"max_finite_abs_error={(actual[finite] - expected[finite]).abs().max().item()}")
print(
    f"negative_infinity_preserved={torch.equal(torch.isneginf(actual), torch.isneginf(expected))}"
)
