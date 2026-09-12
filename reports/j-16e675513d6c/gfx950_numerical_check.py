"""Compare DFlash verify adjustments on GPU with an explicit CPU reference."""

from types import SimpleNamespace

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
)


class _SamplingInfo(SimpleNamespace):
    def __len__(self) -> int:
        return self.batch_size


penalty_cpu = torch.tensor(
    [[0.0, float("-inf"), 2.25], [-3.5, 0.75, 1.0]], dtype=torch.float32
)
bias_cpu = torch.tensor([[1.0, 0.5, -0.25], [0.25, -1.0, 2.0]])
initial_cpu = torch.tensor([[0.5, -1.0, 3.0], [4.0, 2.0, -2.0]]).repeat_interleave(
    4, dim=0
)
sampling_info = _SamplingInfo(
    batch_size=2,
    has_custom_logit_processor=False,
    acc_additive_penalties=penalty_cpu.cuda(),
    penalizer_orchestrator=None,
    grammar_mask=None,
    logit_bias=bias_cpu.cuda(),
)
actual = initial_cpu.cuda()
apply_dflash_verify_logits_adjustments(
    next_token_logits=actual, sampling_info=sampling_info, draft_token_num=4
)

# Independent reference: explicit rows rather than the implementation's broadcast.
expected = initial_cpu.clone()
for batch in range(2):
    for step in range(4):
        expected[batch * 4 + step] += penalty_cpu[batch]
        expected[batch * 4 + step] += bias_cpu[batch]

torch.testing.assert_close(actual.cpu(), expected)
print("device_name=", torch.cuda.get_device_name(0))
print("arch=", torch.cuda.get_device_properties(0).gcnArchName)
print("shape=", tuple(actual.shape), "dtype=", actual.dtype)
print("stop_column_all_negative_inf=", bool(torch.isneginf(actual[:4, 1]).all()))
print(
    "max_finite_abs_error=",
    torch.nan_to_num((actual.cpu() - expected).abs(), nan=0.0, posinf=0.0).max().item(),
)
print("PASS: GPU implementation matches explicit CPU row-loop reference")
