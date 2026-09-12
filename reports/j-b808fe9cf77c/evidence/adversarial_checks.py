from types import SimpleNamespace

import torch

from sglang.srt.speculative.dflash_utils import apply_dflash_verify_logits_adjustments
from sglang.srt.speculative.dspark_components.dspark_verify import (
    verify_logits_adjustments_are_noop,
)


class SamplingInfo(SimpleNamespace):
    def __len__(self):
        return self.batch_size


def info(**overrides):
    values = dict(
        batch_size=1,
        has_custom_logit_processor=False,
        acc_additive_penalties=None,
        penalizer_orchestrator=None,
        grammar_mask=None,
        logit_bias=None,
    )
    values.update(overrides)
    return SamplingInfo(**values)


def main():
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"device={props.name} capability={torch.cuda.get_device_capability(0)}")

    penalties_cpu = torch.tensor(
        [[0.0, float("-inf"), 2.5, -3.0], [1.25, 0.0, -4.0, float("-inf")]],
        dtype=torch.float32,
    )
    bias_cpu = torch.tensor(
        [[0.5, 1.0, -0.5, 2.0], [-0.25, 3.0, 0.5, 4.0]], dtype=torch.float32
    )
    draft = 4
    logits = torch.zeros((2 * draft, 4), dtype=torch.float16, device=device)
    sampling_info = info(
        batch_size=2,
        acc_additive_penalties=penalties_cpu.to(device),
        logit_bias=bias_cpu.to(device),
    )
    apply_dflash_verify_logits_adjustments(
        next_token_logits=logits,
        sampling_info=sampling_info,
        draft_token_num=draft,
    )
    expected = (penalties_cpu + bias_cpu).to(torch.float16)[:, None, :].expand(-1, draft, -1).reshape_as(logits.cpu())
    torch.testing.assert_close(logits.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    assert not verify_logits_adjustments_are_noop(sampling_info)
    assert torch.isneginf(logits.reshape(2, draft, 4)[0, :, 1]).all()
    assert torch.isneginf(logits.reshape(2, draft, 4)[1, :, 3]).all()
    print("gpu_adjustment_matches_explicit_cpu_reference=true")

    for bad_rows, draft_tokens in [(3, 2), (2, 0)]:
        bad = torch.zeros((bad_rows, 4), device=device)
        try:
            apply_dflash_verify_logits_adjustments(
                next_token_logits=bad,
                sampling_info=info(),
                draft_token_num=draft_tokens,
            )
        except ValueError as exc:
            print(f"expected_validation_error={exc}")
        else:
            raise AssertionError("invalid shape/count was accepted")


if __name__ == "__main__":
    main()
