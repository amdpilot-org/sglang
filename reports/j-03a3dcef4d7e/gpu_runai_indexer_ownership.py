"""Focused ownership check for Run:AI tensors retained by the fused DSA loader."""

from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant
from sglang.srt.model_loader.weight_utils import RUNAI_STREAMER_TENSOR_ATTR
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    _load_fused_indexer_wk,
)


def run_case(marked: bool, scale_first: bool) -> tuple[float, int]:
    device = torch.device("cuda", 0)
    source = torch.eye(128, device=device, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    scale = torch.ones((1, 1), device=device, dtype=torch.float32)
    expected = block_quant_dequant(source, scale, [128, 128], torch.bfloat16)
    if marked:
        setattr(source, RUNAI_STREAMER_TENSOR_ATTR, True)
        setattr(scale, RUNAI_STREAMER_TENSOR_ATTR, True)

    name = "model.layers.0.self_attn.indexer.wk.weight"
    fused_name = "model.layers.0.self_attn.indexer.wk_weights_proj.weight"
    fused = torch.nn.Parameter(
        torch.full((130, 128), -1, device=device, dtype=torch.bfloat16)
    )
    pending = {}

    items = [
        (name, source),
        (name.replace("weight", "weight_scale_inv"), scale),
    ]
    if scale_first:
        items.reverse()
    for index, (item_name, tensor) in enumerate(items):
        assert _load_fused_indexer_wk(
            item_name,
            tensor,
            {fused_name: fused},
            pending,
            SimpleNamespace(weight_block_size=[128, 128]),
        )
        if index == 0:
            tensor.zero_()  # Simulate reuse before the paired tensor arrives.

    assert not pending
    error = (fused[:128].float() - expected.float()).abs().max().item()
    return error, torch.count_nonzero(fused[:128]).item()


def main() -> None:
    for scale_first in (False, True):
        marked_error, marked_nonzero = run_case(True, scale_first)
        control_error, control_nonzero = run_case(False, scale_first)
        assert marked_error == 0
        assert control_error > 0
        print(
            f"scale_first={scale_first} marked_max_abs_error={marked_error} "
            f"marked_nonzero={marked_nonzero} control_max_abs_error={control_error} "
            f"control_nonzero={control_nonzero}"
        )
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"arch={torch.cuda.get_device_properties(0).gcnArchName}")


if __name__ == "__main__":
    main()
