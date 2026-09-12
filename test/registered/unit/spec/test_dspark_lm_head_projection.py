from types import SimpleNamespace

import pytest
import torch

from sglang.srt.models.dspark import project_through_lm_head
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class ModelOptFp4LinearMethod:
    def __init__(self, logical_weight: torch.Tensor) -> None:
        self.logical_weight = logical_weight
        self.quant_mode = "w4a16"
        self.calls = 0

    def apply(self, layer, hidden: torch.Tensor, bias):
        self.calls += 1
        assert layer.weight.shape[-1] * 2 == hidden.shape[-1]
        assert bias is None
        return torch.matmul(hidden, self.logical_weight.T)


def test_packed_lm_head_uses_quantized_projection() -> None:
    """Regression for #35354: packed storage is not a dense LM-head matrix."""
    torch.manual_seed(0)
    hidden = torch.randn(5 * 7, 8)
    logical_weight = torch.randn(11, 8)
    quant_method = ModelOptFp4LinearMethod(logical_weight)
    lm_head = SimpleNamespace(
        weight=torch.empty(11, 4, dtype=torch.uint8),
        quant_method=quant_method,
        weight_scale_interleaved=torch.empty(0),
        alpha=torch.empty(0),
        input_size_per_partition=8,
        output_size_per_partition=11,
    )

    # This is the pre-fix DSpark operation, reduced from the reported
    # [245, 5120] x [2560, 248320] shapes while preserving the 2:1 packing.
    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        torch.matmul(hidden, lm_head.weight.T)

    actual = project_through_lm_head(hidden, lm_head)

    assert quant_method.calls == 1
    torch.testing.assert_close(actual, torch.matmul(hidden, logical_weight.T))


def test_dense_lm_head_keeps_matmul_path_and_casts_hidden() -> None:
    weight = torch.randn(7, 4, dtype=torch.float32)
    hidden = torch.randn(3, 4, dtype=torch.float64)
    lm_head = SimpleNamespace(weight=weight, quant_method=None)

    actual = project_through_lm_head(hidden, lm_head)

    assert actual.dtype == weight.dtype
    torch.testing.assert_close(actual, torch.matmul(hidden.float(), weight.T))


class _StaleModelOptFp4LinearMethod(ModelOptFp4LinearMethod):
    def apply(self, *_args, **_kwargs):
        raise AssertionError("a stale quantization method must not be applied")


_StaleModelOptFp4LinearMethod.__name__ = "ModelOptFp4LinearMethod"


def test_stale_modelopt_method_on_dense_weight_keeps_dense_path() -> None:
    weight = torch.randn(6, 4)
    hidden = torch.randn(2, 4)
    lm_head = SimpleNamespace(
        weight=weight,
        quant_method=_StaleModelOptFp4LinearMethod(weight),
    )

    actual = project_through_lm_head(hidden, lm_head)

    torch.testing.assert_close(actual, torch.matmul(hidden, weight.T))
