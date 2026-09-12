"""Regression tests for the ROCm MLA DCP decode return contract."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla_rocm as rocm_mla,
)
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm import (
    DeepseekMLARocmForwardMixin,
    _require_attn_output_and_lse,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRocmDcpDecodeReturnContract(CustomTestCase):
    def _call_actual_forward(self, batch_size):
        backend = mock.Mock(return_value=torch.zeros(batch_size, 2, 8))
        attn = SimpleNamespace(
            current_attention_backend="flashinfer",
            rotary_emb=None,
            kv_lora_rank=8,
            num_local_heads=1,
            attn_mqa_for_dcp_decode=backend,
            _skip_rope_for_dsa_tilelang_fused=lambda: False,
            _fuse_rope_for_trtllm_mla=lambda _forward_batch: False,
        )
        tensor = torch.zeros(batch_size, 8)
        with mock.patch.object(rocm_mla, "is_dcp_mla_decode_phase", return_value=True):
            with self.assertRaisesRegex(RuntimeError, r"requires.*attn_output, lse"):
                DeepseekMLARocmForwardMixin.forward_absorb_rocm_core(
                    attn,
                    tensor,
                    tensor,
                    tensor,
                    tensor,
                    SimpleNamespace(),
                    None,
                    None,
                    None,
                    None,
                )
        return backend

    def test_actual_forward_requests_lse_and_rejects_tensor_batch_one(self):
        backend = self._call_actual_forward(1)
        self.assertIs(backend.call_args.kwargs["return_lse"], True)

    def test_actual_forward_requests_lse_and_rejects_tensor_larger_batch(self):
        backend = self._call_actual_forward(2)
        self.assertIs(backend.call_args.kwargs["return_lse"], True)

    def test_actual_forward_passes_real_lse_to_collective(self):
        for batch_size in (1, 3):
            with self.subTest(batch_size=batch_size):
                output = torch.randn(batch_size, 2, 8)
                lse = torch.randn(batch_size, 2)
                backend = mock.Mock(return_value=(output, lse))
                collective = mock.Mock(
                    side_effect=lambda out, *_args, **_kwargs: out[:, :1]
                )
                attn = SimpleNamespace(
                    current_attention_backend="flashinfer",
                    rotary_emb=None,
                    kv_lora_rank=8,
                    num_local_heads=1,
                    attn_mqa_for_dcp_decode=backend,
                    use_deep_gemm_bmm=False,
                    next_skip_topk=None,
                    o_proj=lambda value: (value, None),
                    _skip_rope_for_dsa_tilelang_fused=lambda: False,
                    _fuse_rope_for_trtllm_mla=lambda _forward_batch: False,
                )
                parallel = SimpleNamespace(
                    attn_dcp_size=2,
                    dcp_comm_backend="a2a",
                    dcp_group=object(),
                )
                tensor = torch.zeros(batch_size, 8)
                with (
                    mock.patch.object(
                        rocm_mla, "is_dcp_mla_decode_phase", return_value=True
                    ),
                    mock.patch.object(rocm_mla, "get_parallel", return_value=parallel),
                    mock.patch.object(
                        rocm_mla, "get_in_autotune_dummy_run", return_value=False
                    ),
                    mock.patch.object(rocm_mla, "dcp_a2a_lse_reduce", collective),
                    mock.patch.object(
                        rocm_mla,
                        "rocm_absorb_v_bmm",
                        side_effect=lambda _self, value: value.flatten(1),
                    ),
                    mock.patch.object(
                        rocm_mla, "is_kv_b_lora_active", return_value=False
                    ),
                ):
                    actual = DeepseekMLARocmForwardMixin.forward_absorb_rocm_core(
                        attn,
                        tensor,
                        tensor,
                        tensor,
                        tensor,
                        SimpleNamespace(),
                        None,
                        None,
                        None,
                        None,
                    )

                self.assertIs(backend.call_args.kwargs["return_lse"], True)
                self.assertIs(collective.call_args.args[1], lse)
                torch.testing.assert_close(actual, output[:, 0])

    def test_tensor_without_lse_fails_clearly_for_batch_one(self):
        result = torch.zeros(1, 4, 8)
        with self.assertRaisesRegex(
            RuntimeError, r"requires.*\(attn_output, lse\).*shape=\(1, 4, 8\)"
        ):
            _require_attn_output_and_lse(result)

    def test_tensor_without_lse_fails_clearly_for_larger_batch(self):
        # A two-row tensor used to unpack successfully and silently treat the
        # second output row as LSE. Validate the container, not its iterable size.
        result = torch.zeros(2, 4, 8)
        with self.assertRaisesRegex(
            RuntimeError, r"requires.*\(attn_output, lse\).*shape=\(2, 4, 8\)"
        ):
            _require_attn_output_and_lse(result)

    def test_tuple_preserves_output_and_lse_for_batch_one_and_larger(self):
        for batch_size in (1, 3):
            with self.subTest(batch_size=batch_size):
                output = torch.randn(batch_size, 4, 8)
                lse = torch.randn(batch_size, 4)
                actual_output, actual_lse = _require_attn_output_and_lse((output, lse))
                self.assertIs(actual_output, output)
                self.assertIs(actual_lse, lse)

    def test_tuple_with_absent_lse_fails_before_collective_use(self):
        with self.assertRaisesRegex(RuntimeError, r"requires.*attn_output, lse"):
            _require_attn_output_and_lse((torch.zeros(2, 4, 8), None))


if __name__ == "__main__":
    unittest.main()
