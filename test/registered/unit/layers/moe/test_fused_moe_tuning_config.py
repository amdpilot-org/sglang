import sys
from pathlib import Path

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    get_config_file_name,
)


TUNER_DIR = (
    Path(__file__).parents[5] / "benchmark" / "kernels" / "fused_moe_triton"
)
sys.path.insert(0, str(TUNER_DIR))

from common_utils import get_config_filename  # noqa: E402


@pytest.mark.parametrize(
    ("num_experts", "shard_intermediate_size"),
    [
        (256, 512),  # Reported Qwen-family shape.
        (8, 28672),  # Independent larger intermediate-size boundary.
    ],
)
def test_int4_tuner_filename_matches_runtime_shape(
    num_experts, shard_intermediate_size
):
    tuner_filename = get_config_filename(
        num_experts,
        shard_intermediate_size,
        hidden_size=4096,
        topk=8,
        dtype=torch.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=True,
        per_channel_quant=False,
        block_shape=None,
    )
    runtime_filename = get_config_file_name(
        num_experts,
        shard_intermediate_size // 2,
        "int4_w4a16",
        block_shape=None,
        per_channel_quant=False,
    )

    assert tuner_filename == runtime_filename


@pytest.mark.parametrize(
    ("quant_flags", "dtype_name"),
    [
        ({}, None),
        ({"use_fp8_w8a8": True}, "fp8_w8a8"),
        ({"use_int8_w8a8": True}, "int8_w8a8"),
        ({"use_int8_w8a16": True}, "int8_w8a16"),
    ],
)
def test_other_tuner_dtype_filenames_remain_runtime_compatible(
    quant_flags, dtype_name
):
    flags = {
        "use_fp8_w8a8": False,
        "use_int8_w8a8": False,
        "use_int8_w8a16": False,
        "use_int4_w4a16": False,
        **quant_flags,
    }
    tuner_filename = get_config_filename(
        64,
        4096,
        hidden_size=4096,
        topk=2,
        dtype=torch.bfloat16,
        per_channel_quant=False,
        block_shape=None,
        **flags,
    )
    runtime_filename = get_config_file_name(
        64,
        2048,
        dtype_name,
        block_shape=None,
        per_channel_quant=False,
    )

    assert tuner_filename == runtime_filename
